package com.movierec.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 大模型输出的**校验器**（阶段四，设计文档 §5.1 输出校验 / R10）。
 *
 * <p>为什么必须有这一层：模型是**不受控的第三方输出**。它会返回候选之外的影片、
 * 自行编造片名、重复同一条、漏掉几条、甚至返回一段自然语言。若不校验就直接落库，
 * 库里会出现"推荐列表里有一部用户根本没被推荐的片"，或者前端显示一句
 * 指向不存在影片的理由 —— 这在答辩现场是致命的。
 *
 * <h3>三条规则（照文档）</h3>
 * <ol>
 *   <li><b>白名单</b>：{@code movieId} 必须落在本次候选集内。候选之外的条目**单独剔除**，
 *       不是整批作废 —— 模型给对 4 条、错 1 条时，没必要连对的一起扔掉。</li>
 *   <li><b>去重</b>：同一 {@code movieId} 只保留第一次出现的那条。</li>
 *   <li><b>顺序</b>：模型返回的顺序**不可信**，一律按候选集原顺序重排输出。</li>
 * </ol>
 * 至于"数量不足时用候选集顺序补齐"（文档原文）—— 补齐需要模板理由，
 * 属于 {@code LlmReasonService} 的职责：本类只做"哪些条可用"的判定，
 * 不生成任何文案。
 *
 * <h3>两处防御性容错</h3>
 * <ul>
 *   <li><b>剥掉 ```json 代码块</b>：实测（2026-09-24）不带 {@code response_format} 时
 *       模型会把 JSON 包进代码块。虽然客户端已经强制要求 JSON 模式，但换厂商/换模型时
 *       未必听话，这里再兜一层，代价只是几行。</li>
 *   <li><b>movieId 允许是字符串</b>：不同模型对 JSON schema 的服从度不同，
 *       {@code "318"} 与 {@code 318} 都接受；其余情况记 malformed。</li>
 * </ul>
 */
@Slf4j
@Component
public class LlmReasonValidator {

    /** 规范的返回结构：{"reasons":[{"movieId":318,"reason":"…"}]} */
    private static final String KEY_REASONS = "reasons";

    private final ObjectMapper objectMapper;

    public LlmReasonValidator(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    /**
     * 校验模型输出。
     *
     * @param rawJson      模型返回的正文（可能带代码块包裹）
     * @param candidateIds 本次候选集的 movieId，**按展示顺序**；既是白名单也是排序依据
     */
    public Result validate(String rawJson, List<Long> candidateIds) {
        Set<Long> whitelist = new LinkedHashSet<>(candidateIds);
        Map<Long, String> accepted = new LinkedHashMap<>();
        int outOfWhitelist = 0;
        int duplicate = 0;
        int blank = 0;
        int malformed = 0;

        JsonNode array = extractArray(rawJson);
        if (array == null) {
            log.warn("大模型输出无法解析为理由数组，整批作废（正文前 120 字：{}）", snippet(rawJson));
            return new Result(Map.of(), 0, 0, 0, 1, true);
        }

        for (JsonNode node : array) {
            long movieId = node.path("movieId").asLong(Long.MIN_VALUE);
            if (movieId == Long.MIN_VALUE || movieId <= 0) {
                malformed++;
                continue;
            }
            if (!whitelist.contains(movieId)) {
                outOfWhitelist++;
                log.warn("大模型返回了候选集之外的影片 {}，该条剔除（白名单共 {} 部）",
                        movieId, whitelist.size());
                continue;
            }
            if (accepted.containsKey(movieId)) {
                duplicate++;
                continue;
            }
            String reason = node.path("reason").asText("").trim();
            if (!StringUtils.hasText(reason)) {
                blank++;
                continue;
            }
            accepted.put(movieId, reason.replaceAll("\\s+", " "));
        }

        // 按候选顺序重排：模型给的顺序不可信，展示顺序必须与推荐列表一致
        Map<Long, String> ordered = new LinkedHashMap<>();
        for (Long id : candidateIds) {
            String reason = accepted.get(id);
            if (reason != null) {
                ordered.put(id, reason);
            }
        }

        Result result = new Result(ordered, outOfWhitelist, duplicate, blank, malformed, false);
        if (result.rejected() > 0) {
            log.warn("大模型输出校验：可用 {} 条，剔除 {} 条（候选外 {} / 重复 {} / 空文案 {} / 字段缺失 {}）",
                    ordered.size(), result.rejected(), outOfWhitelist, duplicate, blank, malformed);
        }
        return result;
    }

    /** 取出理由数组；失败返回 null（由调用方记 parseFailed） */
    private JsonNode extractArray(String rawJson) {
        if (!StringUtils.hasText(rawJson)) {
            return null;
        }
        try {
            JsonNode root = objectMapper.readTree(stripCodeFence(rawJson));
            if (root.isArray()) {
                return root;
            }
            JsonNode reasons = root.path(KEY_REASONS);
            return reasons.isArray() ? reasons : null;
        } catch (Exception e) {
            log.warn("大模型输出不是合法 JSON：{}", e.getMessage());
            return null;
        }
    }

    /** 去掉 ```json … ``` 包裹（实测模型会这么干，见类注释） */
    private static String stripCodeFence(String text) {
        String trimmed = text.trim();
        if (!trimmed.startsWith("```")) {
            return trimmed;
        }
        int firstNewline = trimmed.indexOf('\n');
        if (firstNewline < 0) {
            return trimmed;
        }
        String body = trimmed.substring(firstNewline + 1);
        int fenceEnd = body.lastIndexOf("```");
        return fenceEnd >= 0 ? body.substring(0, fenceEnd).trim() : body.trim();
    }

    private static String snippet(String text) {
        if (text == null) {
            return "(空)";
        }
        String flat = text.replaceAll("\\s+", " ").trim();
        return flat.length() <= 120 ? flat : flat.substring(0, 120) + "…";
    }

    /**
     * 校验结果。
     *
     * @param reasons        可用的理由，**键按候选集顺序**、不含被剔除的条目
     * @param outOfWhitelist 候选集之外的条数
     * @param duplicate      重复的条数
     * @param blank          文案为空的条数
     * @param malformed      movieId 缺失/非法的条数
     * @param parseFailed    整体无法解析（此时 reasons 一定为空）
     */
    public record Result(Map<Long, String> reasons,
                         int outOfWhitelist,
                         int duplicate,
                         int blank,
                         int malformed,
                         boolean parseFailed) {

        /** 被剔除的总条数（论文里"模型输出质量"的量化依据之一） */
        public int rejected() {
            return outOfWhitelist + duplicate + blank + malformed;
        }

        /** 是否还有可用理由；false 时调用方应对全部条目走模板降级 */
        public boolean usable() {
            return !reasons.isEmpty();
        }
    }
}
