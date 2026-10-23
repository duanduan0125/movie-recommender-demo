package com.movierec;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.movierec.common.LlmException;
import com.movierec.config.AppProperties;
import com.movierec.service.impl.LlmClientImpl;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 大模型客户端的单元测试。
 *
 * <p>用 JDK 自带的 {@link HttpServer} 起一个**本地桩服务**，而不是 Mock 掉 HTTP 层 ——
 * 这一组要验的恰恰是 HTTP 层的行为：超时是否真的会抛、5xx 会不会重试、4xx 会不会白重试。
 * Mock 掉就什么都验不到了。
 *
 * <p>三个容易踩的测试陷阱，这里都显式处理了：
 * <ol>
 *   <li>桩服务必须设线程池（{@code setExecutor}）。默认 executor 是单线程串行处理，
 *       超时用例里第二个请求会排在还"睡着"的第一个请求后面，请求计数就不可靠了。</li>
 *   <li>请求计数要**在 handler 入口**统一自增，不要散在各个 handler 里 ——
 *       漏加过一次，结果"客户端明明只发了一次请求"却断言成 0 次。</li>
 *   <li>超时用例里桩服务必须**明显慢于**超时值（这里 3000ms vs 500ms）。
 *       若两者接近，断言会随机翻车。</li>
 * </ol>
 *
 * <p>★ {@link #sendsJsonResponseFormat()} 是一条**回归锁**：实测（2026-09-24，glm-4-flash）
 * 不传 {@code response_format=json_object} 时，模型会把 JSON 包在 ```json 代码块里、
 * 甚至只输出 1 条就当完成，直接解析必然失败。这条用例保证以后没人把这行"看着多余"的参数删掉。
 */
class LlmClientTest {

    private HttpServer server;

    private AppProperties props;

    /** 桩服务收到的请求次数（handler 入口自增） */
    private final AtomicInteger hits = new AtomicInteger();

    /** 最近一次请求的正文，用于断言"发出去的到底是什么" */
    private volatile String lastRequestBody;

    @BeforeEach
    void setUp() {
        props = new AppProperties();
        AppProperties.Llm llm = props.getLlm();
        llm.setApiKey("test-key");
        llm.setModel("stub-model");
        llm.setTimeoutMs(1000);
        llm.setMaxAttempts(2);
    }

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
            server = null;
        }
        hits.set(0);
        lastRequestBody = null;
    }

    @Test
    @DisplayName("正常路径：返回 choices[0].message.content，且只请求一次")
    void returnsContentOnSuccess() {
        startServer(jsonContent("{\"reasons\":[]}"));

        String content = client().completeJson("sys", "user");

        assertThat(content).isEqualTo("{\"reasons\":[]}");
        assertThat(hits.get()).isEqualTo(1);
    }

    @Test
    @DisplayName("★ 必须带 response_format=json_object（实测不带会被包进 ```json 代码块）")
    void sendsJsonResponseFormat() {
        startServer(jsonContent("{}"));

        client().completeJson("sys", "user");

        assertThat(lastRequestBody).contains("\"response_format\"");
        assertThat(lastRequestBody).contains("json_object");
        assertThat(lastRequestBody).contains("stub-model");
    }

    @Test
    @DisplayName("5xx 可重试：第一次 500、第二次 200 → 成功，共请求 2 次")
    void retriesOnceOn5xxThenSucceeds() {
        startServer(exchange -> {
            if (hits.get() == 1) {
                respond(exchange, 500, "{\"error\":\"server busy\"}");
            } else {
                respond(exchange, 200, okBody("{\"reasons\":[]}"));
            }
        });

        String content = client().completeJson("sys", "user");

        assertThat(content).isEqualTo("{\"reasons\":[]}");
        assertThat(hits.get()).isEqualTo(2);
    }

    @Test
    @DisplayName("★ 4xx 不重试：401 立即失败，只请求一次（重试也没用，只是白烧配额）")
    void doesNotRetryOn4xx() {
        startServer(exchange -> respond(exchange, 401, "{\"error\":\"invalid api key\"}"));

        assertThatThrownBy(() -> client().completeJson("sys", "user"))
                .isInstanceOf(LlmException.class)
                .hasMessageContaining("401");

        assertThat(hits.get()).isEqualTo(1);
    }

    @Test
    @DisplayName("★ 超时可重试：桩服务慢 3 秒、超时 500ms → 抛异常且确实重试了一次")
    void retriesOnTimeout() {
        startServer(slowHandler(3000L));
        props.getLlm().setTimeoutMs(500);

        long start = System.currentTimeMillis();
        assertThatThrownBy(() -> client().completeJson("sys", "user"))
                .isInstanceOf(LlmException.class);
        long elapsed = System.currentTimeMillis() - start;

        // 两次请求都到达了桩服务：500ms 超时 + 500ms 退避 + 500ms 超时 ≈ 1500ms
        assertThat(hits.get()).isEqualTo(2);
        assertThat(elapsed).as("耗时 %d ms，应包含「超时 + 退避 + 再超时」三段", elapsed)
                .isGreaterThanOrEqualTo(1200L);
    }

    @Test
    @DisplayName("maxAttempts=1 时不重试：超时后只请求一次")
    void noRetryWhenMaxAttemptsIsOne() {
        startServer(slowHandler(3000L));
        props.getLlm().setTimeoutMs(500);
        props.getLlm().setMaxAttempts(1);

        assertThatThrownBy(() -> client().completeJson("sys", "user"))
                .isInstanceOf(LlmException.class);

        assertThat(hits.get()).isEqualTo(1);
    }

    @Test
    @DisplayName("响应不是 JSON 时抛 LlmException，而不是把脏数据往下传")
    void throwsWhenResponseIsNotJson() {
        startServer(exchange -> respond(exchange, 200, "<html>not json</html>"));

        assertThatThrownBy(() -> client().completeJson("sys", "user"))
                .isInstanceOf(LlmException.class)
                .hasMessageContaining("不是合法 JSON");
    }

    @Test
    @DisplayName("正文为空时抛 LlmException（模型偶尔会返回空 content）")
    void throwsWhenContentIsBlank() {
        startServer(jsonContent(""));

        assertThatThrownBy(() -> client().completeJson("sys", "user"))
                .isInstanceOf(LlmException.class)
                .hasMessageContaining("未返回正文");
    }

    @Test
    @DisplayName("配置不完整时立刻失败，且不发出任何请求（而不是发一个必然失败的请求）")
    void throwsWhenConfigMissing() {
        // 注意顺序：startServer 会写入 baseUrl，所以必须"先启动、后清空"
        startServer(jsonContent("{}"));
        props.getLlm().setBaseUrl("");

        assertThatThrownBy(() -> client().completeJson("sys", "user"))
                .isInstanceOf(LlmException.class)
                .hasMessageContaining("配置不完整");

        assertThat(hits.get()).isZero();
    }

    // ------------------------------------------------------------------ 测试脚手架

    private LlmClientImpl client() {
        return new LlmClientImpl(props, new ObjectMapper());
    }

    /**
     * 启动桩服务。所有 handler 都经过 {@link #count(HttpHandler)} 包一层，
     * 保证"请求到达"这件事在入口处就被记下来。
     */
    private void startServer(HttpHandler handler) {
        try {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.setExecutor(Executors.newFixedThreadPool(4));
            server.createContext("/v1/chat/completions", count(handler));
            server.start();
        } catch (IOException e) {
            throw new IllegalStateException("无法启动本地桩服务", e);
        }
        props.getLlm().setBaseUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/v1");
    }

    private HttpHandler count(HttpHandler inner) {
        return exchange -> {
            hits.incrementAndGet();
            inner.handle(exchange);
        };
    }

    private HttpHandler jsonContent(String content) {
        return exchange -> respond(exchange, 200, okBody(content));
    }

    /** 先睡 delayMs 再回 200：用于制造确定性超时 */
    private HttpHandler slowHandler(long delayMs) {
        return exchange -> {
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            respond(exchange, 200, okBody("{}"));
        };
    }

    private String okBody(String content) {
        return "{\"choices\":[{\"message\":{\"content\":"
                + new ObjectMapper().valueToTree(content).toString()
                + "}}],\"usage\":{\"prompt_tokens\":137,\"completion_tokens\":81}}";
    }

    private void respond(HttpExchange exchange, int status, String body) throws IOException {
        try {
            lastRequestBody = new String(exchange.getRequestBody().readAllBytes(),
                    StandardCharsets.UTF_8);
        } catch (IOException ignored) {
            lastRequestBody = null;
        }
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
        try {
            exchange.sendResponseHeaders(status, bytes.length);
            try (var out = exchange.getResponseBody()) {
                out.write(bytes);
            }
        } catch (IOException e) {
            // 客户端已因超时断开时会出现，属预期，不影响用例结论
            exchange.close();
        }
    }
}
