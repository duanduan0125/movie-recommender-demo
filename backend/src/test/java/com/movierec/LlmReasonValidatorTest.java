package com.movierec;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.movierec.service.LlmReasonValidator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 大模型输出校验器的单元测试（设计文档 §5.1 输出校验 / R10）。
 *
 * <p>这些用例模拟的都是**真实可能出现的坏输出**：候选之外的影片、重复条目、
 * 空文案、字段缺失、非法 JSON、被 ```json 包裹。它们不是"为了凑覆盖率"—— R10 之所以
 * 被写进修订清单，就是因为原方案没有这一层校验。
 */
class LlmReasonValidatorTest {

    /** 候选集（与推荐接口的展示顺序一致） */
    private static final List<Long> CANDIDATES = List.of(318L, 1L, 260L, 2571L, 296L);

    private LlmReasonValidator validator;

    @BeforeEach
    void setUp() {
        validator = new LlmReasonValidator(new ObjectMapper());
    }

    @Test
    @DisplayName("正常输出：5 条全部可用，键按候选顺序")
    void acceptsWellFormedOutput() {
        String json = """
                {"reasons":[
                  {"movieId":318,"reason":"因为你喜欢《肖申克的救赎》"},
                  {"movieId":1,"reason":"同为成长题材"},
                  {"movieId":260,"reason":"史诗科幻"},
                  {"movieId":2571,"reason":"科幻经典"},
                  {"movieId":296,"reason":"真实事件改编"}
                ]}""";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.usable()).isTrue();
        assertThat(r.parseFailed()).isFalse();
        assertThat(r.rejected()).isZero();
        assertThat(r.reasons()).hasSize(5);
        assertThat(r.reasons().keySet()).containsExactlyElementsOf(CANDIDATES);
    }

    @Test
    @DisplayName("★ 候选之外的影片被单独剔除，其余保留（不是整批作废）")
    void rejectsMovieOutsideWhitelistButKeepsOthers() {
        String json = """
                {"reasons":[
                  {"movieId":318,"reason":"理由一"},
                  {"movieId":999999,"reason":"这部不在候选里"},
                  {"movieId":1,"reason":"理由二"}
                ]}""";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.outOfWhitelist()).isEqualTo(1);
        assertThat(r.rejected()).isEqualTo(1);
        assertThat(r.reasons()).containsOnlyKeys(318L, 1L);
        // 白名单外的条目不得出现在结果里
        assertThat(r.reasons()).doesNotContainKey(999999L);
    }

    @Test
    @DisplayName("模型返回的顺序不可信：一律按候选顺序重排")
    void reordersByCandidateOrder() {
        String json = """
                {"reasons":[
                  {"movieId":296,"reason":"C"},
                  {"movieId":318,"reason":"A"},
                  {"movieId":260,"reason":"B"}
                ]}""";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.reasons().keySet()).containsExactly(318L, 260L, 296L);
    }

    @Test
    @DisplayName("重复条目只保留第一次出现的那条")
    void keepsFirstOccurrenceOfDuplicate() {
        String json = """
                {"reasons":[
                  {"movieId":318,"reason":"第一次"},
                  {"movieId":318,"reason":"第二次"}
                ]}""";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.duplicate()).isEqualTo(1);
        assertThat(r.reasons()).containsEntry(318L, "第一次");
    }

    @Test
    @DisplayName("文案为空的条目被剔除（不能让前端显示一张没有理由的卡）")
    void rejectsBlankReason() {
        String json = """
                {"reasons":[
                  {"movieId":318,"reason":"   "},
                  {"movieId":1,"reason":"有用"}
                ]}""";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.blank()).isEqualTo(1);
        assertThat(r.reasons()).containsOnlyKeys(1L);
    }

    @Test
    @DisplayName("movieId 缺失或非法的条目记 malformed")
    void rejectsMalformedEntries() {
        String json = """
                {"reasons":[
                  {"reason":"没有 movieId"},
                  {"movieId":0,"reason":"id 非法"},
                  {"movieId":2571,"reason":"这一条有效（2571 在候选集内）"}
                ]}""";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.malformed()).isEqualTo(2);
        assertThat(r.reasons()).containsOnlyKeys(2571L);
    }

    @Test
    @DisplayName("movieId 是字符串也能接受（不同模型对 schema 的服从度不同）")
    void acceptsStringMovieId() {
        String json = "{\"reasons\":[{\"movieId\":\"318\",\"reason\":\"字符串 id\"}]}";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.malformed()).isZero();
        assertThat(r.reasons()).containsEntry(318L, "字符串 id");
    }

    @Test
    @DisplayName("★ 被 ```json 代码块包裹时仍能解析（实测模型会这么干）")
    void toleratesCodeFence() {
        String json = """
                ```json
                {"reasons":[{"movieId":318,"reason":"代码块里的 JSON"}]}
                ```""";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.parseFailed()).isFalse();
        assertThat(r.reasons()).containsEntry(318L, "代码块里的 JSON");
    }

    @Test
    @DisplayName("顶层直接是数组也接受（容错，不额外要求外层包装）")
    void acceptsTopLevelArray() {
        String json = "[{\"movieId\":318,\"reason\":\"裸数组\"}]";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.parseFailed()).isFalse();
        assertThat(r.reasons()).containsEntry(318L, "裸数组");
    }

    @Test
    @DisplayName("★ 不是 JSON（或缺少 reasons 键）时整批作废，usable=false 交给降级")
    void marksParseFailure() {
        LlmReasonValidator.Result notJson = validator.validate("抱歉，我无法完成这个请求。", CANDIDATES);
        assertThat(notJson.parseFailed()).isTrue();
        assertThat(notJson.usable()).isFalse();
        assertThat(notJson.rejected()).isEqualTo(1);

        LlmReasonValidator.Result wrongShape = validator.validate("{\"foo\":[]}", CANDIDATES);
        assertThat(wrongShape.parseFailed()).isTrue();
        assertThat(wrongShape.usable()).isFalse();
    }

    @Test
    @DisplayName("数量不足不算失败：可用的几条先用，缺口由调用方按候选顺序补齐")
    void shortageIsNotFailure() {
        String json = "{\"reasons\":[{\"movieId\":318,\"reason\":\"只有一条\"}]}";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        assertThat(r.usable()).isTrue();
        assertThat(r.parseFailed()).isFalse();
        assertThat(r.reasons()).hasSize(1);
    }

    @Test
    @DisplayName("空输入与 null 输入都不抛异常，只记作解析失败")
    void handlesBlankInput() {
        for (String raw : new String[]{null, "", "   "}) {
            LlmReasonValidator.Result r = validator.validate(raw, CANDIDATES);
            assertThat(r.parseFailed()).as("输入 %s", raw).isTrue();
            assertThat(r.usable()).isFalse();
        }
    }

    @Test
    @DisplayName("文案里的换行被压成空格（避免落库后前端排版错乱）")
    void collapsesWhitespaceInReason() {
        String json = "{\"reasons\":[{\"movieId\":318,\"reason\":\"第一行\\n第二行\\t第三行\"}]}";

        LlmReasonValidator.Result r = validator.validate(json, CANDIDATES);

        Map<Long, String> reasons = r.reasons();
        assertThat(reasons.get(318L)).isEqualTo("第一行 第二行 第三行");
    }
}
