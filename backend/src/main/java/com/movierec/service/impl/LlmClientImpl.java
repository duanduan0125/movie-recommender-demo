package com.movierec.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.movierec.common.LlmException;
import com.movierec.config.AppProperties;
import com.movierec.service.LlmClient;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * OpenAI 兼容协议的大模型客户端（阶段四，设计文档 §5.1）。
 *
 * <h3>重试口径（照文档实现）</h3>
 * "仅对超时与 5xx 重试 1 次；4xx 不重试（重试也没用）"：
 * <ul>
 *   <li>超时 / 连接异常（{@link ResourceAccessException}）→ 可重试</li>
 *   <li>5xx（{@link RestClientResponseException} 且状态码 5xx）→ 可重试</li>
 *   <li>4xx → **立即失败**。401/403 是 Key 错、404 是 base-url 或模型名错，
 *       重试只是白烧配额并让用户多等一轮超时</li>
 * </ul>
 *
 * <h3>★ 两处用实测数据定下来的实现细节</h3>
 * <ol>
 *   <li><b>必须带 {@code response_format={"type":"json_object"}}</b>：
 *       实测（2026-09-24，glm-4-flash）不带的两次里，一次把 JSON 包在 ```json 代码块中
 *       且只输出 1 条理由（completion_tokens=34），一次 6.49 秒后返回的内容
 *       完全无法解析。带上之后 4 次全部返回纯 JSON 且 5 条齐全。</li>
 *   <li><b>超时取 12 秒而不是文档写的 5 秒</b>：同一批实测的 4 次耗时是
 *       6.49 / 8.14 / 5.02 / 6.71 秒（均值 6.6 秒）。按 5 秒实现等于每次调用都超时降级，
 *       功能等于没接。详见 {@link AppProperties.Llm#getTimeoutMs()}。</li>
 * </ol>
 *
 * <p>日志里**只打模型名、耗时、token 数**，绝不打 api-key 与请求头。
 */
@Slf4j
@Service
public class LlmClientImpl implements LlmClient {

    /** 重试前的固定退避，避免上游刚抖动就立刻重发 */
    private static final long RETRY_BACKOFF_MS = 500L;

    /** 连接超时（与读超时分开：连不上通常几秒内就有结论，不必等满读超时） */
    private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(5);

    /** 异常信息里截断响应体，避免把整页 HTML 灌进日志 */
    private static final int BODY_SNIPPET_MAX = 200;

    private final AppProperties props;

    private final ObjectMapper objectMapper;

    private final RestClient restClient;

    public LlmClientImpl(AppProperties props, ObjectMapper objectMapper) {
        this.props = props;
        this.objectMapper = objectMapper;
        this.restClient = buildRestClient(props.getLlm());
    }

    private static RestClient buildRestClient(AppProperties.Llm llm) {
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(CONNECT_TIMEOUT)
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory(httpClient);
        // ★ 这里只兜"非正数"的底（避免配成 0 导致请求瞬间超时），**不做别的下限保护**。
        //   曾经写成 Math.max(timeoutMs, 1000) 当作"保护值"，结果是：配置的任何小于 1 秒的
        //   超时都被静默抬到 1 秒，超时形同虚设 —— 单测量出来的现象是"300ms 超时下、
        //   桩服务慢 800ms，调用却成功返回"。隐式兜底比不做兜底更危险：它让配置说谎。
        factory.setReadTimeout(Duration.ofMillis(Math.max(llm.getTimeoutMs(), 100L)));
        return RestClient.builder()
                .requestFactory(factory)
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    @Override
    public String completeJson(String systemPrompt, String userPrompt) {
        AppProperties.Llm llm = props.getLlm();
        if (!StringUtils.hasText(llm.getBaseUrl())
                || !StringUtils.hasText(llm.getApiKey())
                || !StringUtils.hasText(llm.getModel())) {
            throw new LlmException("大模型配置不完整：application-local.yml 的 app.llm 需要 "
                    + "base-url / api-key / model 三项");
        }

        String url = llm.getBaseUrl().replaceAll("/+$", "") + "/chat/completions";
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("model", llm.getModel());
        payload.put("messages", List.of(
                Map.of("role", "system", "content", systemPrompt),
                Map.of("role", "user", "content", userPrompt)));
        payload.put("temperature", llm.getTemperature());
        payload.put("max_tokens", llm.getMaxTokens());
        payload.put("response_format", Map.of("type", "json_object"));

        int attempts = Math.max(llm.getMaxAttempts(), 1);
        LlmException last = null;
        for (int attempt = 1; attempt <= attempts; attempt++) {
            long start = System.currentTimeMillis();
            try {
                String raw = restClient.post()
                        .uri(url)
                        .header(HttpHeaders.AUTHORIZATION, "Bearer " + llm.getApiKey())
                        .body(payload)
                        .retrieve()
                        .body(String.class);
                return extractContent(raw, llm, System.currentTimeMillis() - start);
            } catch (RestClientResponseException e) {
                boolean retryable = e.getStatusCode().is5xxServerError();
                last = new LlmException("HTTP " + e.getStatusCode().value() + "：" + snippet(e.getResponseBodyAsString()), e);
                log.warn("大模型调用失败（HTTP {}，第 {}/{} 次，可重试={}）：{}",
                        e.getStatusCode().value(), attempt, attempts, retryable, snippet(e.getResponseBodyAsString()));
                if (!retryable) {
                    throw last;
                }
            } catch (ResourceAccessException e) {
                last = new LlmException("调用超时或连接异常：" + e.getMessage(), e);
                log.warn("大模型调用失败（超时/连接异常，第 {}/{} 次，耗时 {} ms）：{}",
                        attempt, attempts, System.currentTimeMillis() - start, e.getMessage());
            }
            sleepBeforeRetry(attempt, attempts);
        }
        throw last == null ? new LlmException("大模型调用失败（未捕获到具体原因）") : last;
    }

    private String extractContent(String raw, AppProperties.Llm llm, long elapsedMs) {
        if (!StringUtils.hasText(raw)) {
            throw new LlmException("大模型响应体为空");
        }
        JsonNode root;
        try {
            root = objectMapper.readTree(raw);
        } catch (Exception e) {
            throw new LlmException("大模型响应不是合法 JSON：" + snippet(raw), e);
        }
        String content = root.path("choices").path(0).path("message").path("content").asText("");
        if (!StringUtils.hasText(content)) {
            throw new LlmException("大模型未返回正文（choices[0].message.content 为空）：" + snippet(raw));
        }
        log.debug("大模型调用成功：model={} 耗时={} ms promptTokens={} completionTokens={}",
                llm.getModel(), elapsedMs,
                root.path("usage").path("prompt_tokens").asInt(-1),
                root.path("usage").path("completion_tokens").asInt(-1));
        return content;
    }

    private void sleepBeforeRetry(int attempt, int attempts) {
        if (attempt >= attempts) {
            return;
        }
        try {
            Thread.sleep(RETRY_BACKOFF_MS);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new LlmException("等待重试时被中断", ie);
        }
    }

    private static String snippet(String text) {
        if (text == null) {
            return "(空)";
        }
        String flat = text.replaceAll("\\s+", " ").trim();
        return flat.length() <= BODY_SNIPPET_MAX ? flat : flat.substring(0, BODY_SNIPPET_MAX) + "…";
    }
}
