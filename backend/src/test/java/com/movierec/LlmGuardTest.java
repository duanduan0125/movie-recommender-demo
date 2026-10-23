package com.movierec;

import com.movierec.config.AppProperties;
import com.movierec.service.LlmGuard;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 大模型闸门（开关 / 熔断 / 当日上限）的单元测试。
 *
 * <p>这一组用不到网络与 Spring 容器：{@link LlmGuard} 的输入就是配置与"成功/失败"事件，
 * 正好可以把三条策略的边界精确钉住 —— 尤其是"连续失败到阈值才熔断、成功后计数归零"，
 * 在真实环境里靠人工触发代价很高。
 */
class LlmGuardTest {

    private AppProperties props;

    private LlmGuard guard;

    @BeforeEach
    void setUp() {
        props = new AppProperties();
        AppProperties.Llm llm = props.getLlm();
        llm.setEnabled(true);
        llm.setCircuitFailureThreshold(3);
        llm.setCircuitCooldownMs(150L);
        llm.setDailyCallLimit(2);
        guard = new LlmGuard(props);
    }

    @Test
    @DisplayName("总开关关闭时一律不放行（单测与验收脚本靠这条不依赖外网）")
    void disabledBlocksAllCalls() {
        props.getLlm().setEnabled(false);

        assertThat(guard.allowCall()).isFalse();
        assertThat(guard.isCircuitOpen()).isFalse();
        assertThat(guard.callsToday()).isZero();
    }

    @Test
    @DisplayName("未达阈值不熔断：连续失败 2 次（阈值 3）仍放行")
    void belowThresholdKeepsCircuitClosed() {
        guard.recordFailure();
        guard.recordFailure();

        assertThat(guard.consecutiveFailures()).isEqualTo(2);
        assertThat(guard.isCircuitOpen()).isFalse();
        assertThat(guard.allowCall()).isTrue();
    }

    @Test
    @DisplayName("★ 连续失败达阈值即熔断，冷却期内所有请求直接降级")
    void reachingThresholdOpensCircuit() {
        guard.recordFailure();
        guard.recordFailure();
        guard.recordFailure();

        assertThat(guard.isCircuitOpen()).isTrue();
        assertThat(guard.allowCall()).isFalse();
    }

    @Test
    @DisplayName("一次成功即把连续失败计数清零（偶发抖动不会攒成熔断）")
    void successResetsFailureStreak() {
        guard.recordFailure();
        guard.recordFailure();
        guard.recordSuccess();
        guard.recordFailure();

        assertThat(guard.consecutiveFailures()).isEqualTo(1);
        assertThat(guard.isCircuitOpen()).isFalse();
    }

    @Test
    @DisplayName("冷却时间一到自动放行（半开试探），不必依赖定时任务")
    void cooldownReleasesCircuit() throws InterruptedException {
        guard.recordFailure();
        guard.recordFailure();
        guard.recordFailure();
        assertThat(guard.allowCall()).isFalse();

        Thread.sleep(200L);

        assertThat(guard.isCircuitOpen()).isFalse();
        assertThat(guard.allowCall()).isTrue();
    }

    @Test
    @DisplayName("当日调用上限：达到上限后不再放行（超限即走模板降级）")
    void dailyLimitBlocksFurtherCalls() {
        assertThat(guard.allowCall()).isTrue();
        assertThat(guard.allowCall()).isTrue();

        assertThat(guard.allowCall()).isFalse();
        assertThat(guard.callsToday()).isEqualTo(2);
    }

    @Test
    @DisplayName("reset 把熔断与计数一起归零（批量预生成按批重置时用）")
    void resetClearsCircuitAndCounters() {
        guard.allowCall();
        guard.recordFailure();
        guard.recordFailure();
        guard.recordFailure();

        guard.reset();

        assertThat(guard.isCircuitOpen()).isFalse();
        assertThat(guard.consecutiveFailures()).isZero();
        assertThat(guard.callsToday()).isZero();
        assertThat(guard.allowCall()).isTrue();
    }
}
