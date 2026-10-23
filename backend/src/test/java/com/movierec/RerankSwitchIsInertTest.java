package com.movierec;

import com.movierec.service.LlmReasonService;
import com.movierec.vo.RecommendItemVO;
import com.movierec.vo.RecommendVO;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * REVIEW-03 补的守门断言：**误开 {@code app.llm.rerank-enabled=true} 也不改变行为**。
 *
 * <p>R16 把重排降级为"可选、默认关闭"，但 Java 侧并没有接这条路径（依据 EXP-03：
 * 采样口径下重排使 HR@10 由 0.6520 降到 0.4200）。于是这个开关是"改了不生效"的 ——
 * 与其静默，不如用测试把它钉成显式事实：
 * <ol>
 *   <li>上下文能正常启动（同时覆盖 {@code LlmReasonServiceImpl#warnIfRerankEnabled}
 *       这条启动检查，确保它只是打日志、不会让应用起不来）；</li>
 *   <li>即使开关打开，{@code attachReasons} 也**不改变推荐顺序** ——
 *       线上顺序始终直接取召回结果（这是 R16 的原文要求）。</li>
 * </ol>
 */
@SpringBootTest(properties = {"app.llm.enabled=false", "app.llm.rerank-enabled=true"})
class RerankSwitchIsInertTest {

    @Autowired
    private LlmReasonService llmReasonService;

    @Test
    @DisplayName("★ rerank-enabled=true 时推荐顺序原样、理由照挂（只多一条启动 WARN）")
    void rerankSwitchDoesNotAffectOnlinePath() {
        RecommendVO vo = new RecommendVO();
        vo.setSource("rec_result");
        vo.setSourceLabel("离线模型推荐");
        List<RecommendItemVO> items = new ArrayList<>();
        for (long movieId : new long[]{318L, 1L, 260L, 2571L, 296L, 527L}) {
            RecommendItemVO item = new RecommendItemVO();
            item.setMovieId(movieId);
            item.setTitle("movie-" + movieId);
            item.setRatingAvg(new BigDecimal("4.5"));
            item.setRatingCount(1000);
            items.add(item);
        }
        vo.setItems(items);
        vo.setSize(items.size());

        llmReasonService.attachReasons(1L, vo);

        assertThat(vo.getItems()).extracting(RecommendItemVO::getMovieId)
                .as("重排开关不得改变顺序（R16：线上顺序直接取召回结果）")
                .containsExactly(318L, 1L, 260L, 2571L, 296L, 527L);
        assertThat(vo.getSource()).isEqualTo("rec_result");
        assertThat(vo.getItems().subList(0, 5)).allSatisfy(item ->
                assertThat(item.getReason()).isNotBlank());
        assertThat(vo.getItems().get(5).getReason()).isNull();
    }
}
