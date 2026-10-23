package com.movierec.service;

import com.movierec.config.AppProperties;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 大模型调用的**闸门**：总开关、熔断、当日调用上限（设计文档 §5.1、§5.4）。
 *
 * <p>为什么把这三件事单独拿出来，而不是塞在客户端里：
 * <ol>
 *   <li>客户端只负责"把一次请求发出去、把响应拿回来"，它失败时**必须有人决定**
 *       "还要不要再试、后面的请求还发不发" —— 那是业务侧的策略，不是 HTTP 细节；</li>
 *   <li>闸门的状态是**跨请求**的，要能单独测。</li>
 * </ol>
 *
 * <h3>★ 关于"批次"的一处取舍（本项目的决定，与文档措辞不同）</h3>
 * 设计文档 §5.1 写的是"同一批次内连续失败 5 次后，本批次后续请求直接走降级"。
 * 但在线路径**每次接口调用只发 1 次大模型请求**（一次调用产出全部理由），
 * "批次内连续 5 次失败"在在线路径上**永远无法触发** —— 那个口径只对
 * 批量预生成（阶段七 §7.3，一个用户一次调用）有意义。
 *
 * <p>若严格按批次实现，在线路径就只剩"单次失败即降级"，
 * 而"避免整页卡住"这个初衷反而落不了地：上游故障时，**每一个**用户请求
 * 都会先白等一次超时。故这里实现为**应用级计数**（连续失败达阈值即熔断，
 * 冷却期内所有请求直接降级），并在代码里显式提供 {@link #reset()} 供批量任务
 * 按批重置 —— 两种语义都保住了。该取舍已记入实验记录 TEST-08。
 */
@Slf4j
@Component
public class LlmGuard {

    private final AppProperties props;

    /** 连续失败次数；任一次成功即归零 */
    private final AtomicInteger consecutiveFailures = new AtomicInteger();

    /** 熔断开始时刻（毫秒）；0 表示当前没有熔断 */
    private volatile long circuitOpenedAt = 0L;

    /** 当日已发起的调用次数 */
    private final AtomicInteger callsToday = new AtomicInteger();

    /** callsToday 对应的日期，跨日时归零 */
    private volatile LocalDate counterDate = LocalDate.now();

    public LlmGuard(AppProperties props) {
        this.props = props;
    }

    /**
     * 是否放行本次调用。
     *
     * <p>返回 {@code false} 时调用方**直接走模板降级**，不要再发起调用。
     * 注意：放行时会占用一个当日额度（统计的是"尝试发起"的次数，含失败）。
     */
    public boolean allowCall() {
        AppProperties.Llm llm = props.getLlm();
        if (!llm.isEnabled()) {
            log.debug("大模型开关已关闭（app.llm.enabled=false），走模板降级");
            return false;
        }
        if (isCircuitOpen()) {
            log.debug("大模型处于熔断冷却期，直接降级");
            return false;
        }
        rolloverIfNeeded();
        int used = callsToday.get();
        if (used >= llm.getDailyCallLimit()) {
            log.warn("大模型当日调用已达上限 {}（已用 {}），本次直接降级", llm.getDailyCallLimit(), used);
            return false;
        }
        callsToday.incrementAndGet();
        return true;
    }

    /**
     * 当前是否处于熔断冷却期。
     *
     * <p>冷却时间一到就自动放行下一次调用（半开试探）：成功则彻底恢复，
     * 失败则重新进入熔断 —— 不需要额外的定时任务。
     *
     * <p>★ 注意：本方法**有副作用**（冷却结束时会把熔断状态与失败计数复位），
     * 所以它不是纯查询，别用它做"只看不动"的状态展示。改名的代价是
     * 调用点语义变模糊，这里选择用注释说明（REVIEW-04 记录）。
     */
    public boolean isCircuitOpen() {
        long openedAt = circuitOpenedAt;
        if (openedAt == 0L) {
            return false;
        }
        long cooldown = props.getLlm().getCircuitCooldownMs();
        if (System.currentTimeMillis() - openedAt >= cooldown) {
            circuitOpenedAt = 0L;
            consecutiveFailures.set(0);
            log.info("大模型熔断冷却结束，放行下一次调用做试探");
            return false;
        }
        return true;
    }

    /** 调用成功：清零连续失败计数 */
    public void recordSuccess() {
        consecutiveFailures.set(0);
        circuitOpenedAt = 0L;
    }

    /** 调用失败：累计连续失败，达阈值即进入熔断 */
    public void recordFailure() {
        int failures = consecutiveFailures.incrementAndGet();
        if (failures >= props.getLlm().getCircuitFailureThreshold() && circuitOpenedAt == 0L) {
            circuitOpenedAt = System.currentTimeMillis();
            log.warn("大模型连续失败 {} 次，进入熔断：冷却 {} 秒内所有调用直接走模板降级",
                    failures, props.getLlm().getCircuitCooldownMs() / 1000);
        }
    }

    /** 连续失败次数（供测试与状态查询） */
    public int consecutiveFailures() {
        return consecutiveFailures.get();
    }

    /** 当日已发起的调用次数（供测试与成本统计）。同样带跨日归零的副作用。 */
    public int callsToday() {
        rolloverIfNeeded();
        return callsToday.get();
    }

    /** 归零：熔断状态、连续失败计数、当日计数。批量预生成脚本按批重置时用。 */
    public void reset() {
        consecutiveFailures.set(0);
        circuitOpenedAt = 0L;
        callsToday.set(0);
        counterDate = LocalDate.now();
    }

    private void rolloverIfNeeded() {
        LocalDate today = LocalDate.now();
        if (!today.equals(counterDate)) {
            counterDate = today;
            callsToday.set(0);
            log.info("跨日，大模型当日调用计数已归零");
        }
    }
}
