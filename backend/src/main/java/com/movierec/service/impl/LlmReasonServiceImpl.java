package com.movierec.service.impl;

import com.movierec.common.LlmException;
import com.movierec.config.AppProperties;
import com.movierec.entity.LlmReason;
import com.movierec.mapper.LlmReasonMapper;
import com.movierec.mapper.RatingMapper;
import com.movierec.mapper.RecommendMapper;
import com.movierec.mapper.UserPreferenceMapper;
import com.movierec.service.LlmClient;
import com.movierec.service.LlmGuard;
import com.movierec.service.LlmPromptTemplates;
import com.movierec.service.LlmReasonService;
import com.movierec.service.LlmReasonValidator;
import com.movierec.vo.MyRatingVO;
import com.movierec.vo.AlsoLikedVO;
import com.movierec.vo.RecommendItemVO;
import com.movierec.vo.RecommendVO;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.time.LocalDate;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 推荐理由服务实现（阶段四 §5.1–§5.4）。
 *
 * <h3>一次请求的完整流程</h3>
 * <pre>
 * 前 N 条
 *   ├─ 开关关闭（单测 / 验收环境）→ 模板理由，不联网、不落库
 *   ├─ 查 llm_reason（用户 + 批次 + 这 N 部）命中 → 直接用，不调用大模型
 *   └─ 未命中的部分
 *        ├─ 闸门不放行（熔断中 / 超当日上限）→ 模板理由，落库 source=1
 *        └─ 调用一次大模型（一次调用产出全部理由）
 *             ├─ 成功 → 白名单/去重校验 → 可用的落库 source=0
 *             └─ 失败或输出不可用 → 模板理由，落库 source=1
 * </pre>
 *
 * <h3>★ 三处需要说明的判断（都写在这里，免得以后被当成"随手写的"）</h3>
 * <ol>
 *   <li><b>缓存命中不区分 source</b>：同一 (用户, 批次) 下已有理由就直接用，
 *       哪怕它是上次降级留下的模板理由。依据是设计文档 §5.3 的原文——"按
 *       (user_id, batch_date) 命中缓存即不再调用"，以及它的失效机制是本批次整体删除。
 *       代价：降级理由会一直服务到批次变更（离线路径是重训，兜底路径是次日），
 *       换来的是"断网演示时不会每次请求都白等一次超时"。想立即刷新就删该批次的旧行。</li>
 *   <li><b>开关关闭时既不联网也不落库</b>：{@code app.llm.enabled=false} 是
 *       "这个环境不用大模型"的显式声明（单元测试、HTTP 验收脚本）。此时往 llm_reason
 *       写模板理由只会污染论文要用的降级率统计，所以直接就地生成、不落库。</li>
 *   <li><b>白名单只认"本次要求生成的影片"</b>：候选上下文喂 20 部（帮助模型写出具体理由），
 *       但要求它只对这 N 部返回；因此校验的白名单就是这 N 部。
 *       模型若多返回候选池里的其它影片，会被校验剔除并记入日志（可视作模型质量问题）。</li>
 * </ol>
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class LlmReasonServiceImpl implements LlmReasonService {

    /** 取用户高分影片的条数：作为"为什么推荐给你"的上下文，5 条足够且省 token */
    private static final int HISTORY_LIMIT = 5;

    /**
     * 找协同依据时，**每部"用户喜欢的影片"**取多少条共现邻居来比对候选。
     *
     * <p>取 100 是权衡：用户喜欢的片可能很冷门，邻居榜不长，取全也够用；
     * 太大则会引入"只被一两个人共同喜欢"的弱证据（那是噪音，写进理由反而不可信）。
     */
    private static final int HINT_NEIGHBOURS = 100;

    private final AppProperties props;

    private final LlmClient llmClient;

    private final LlmGuard llmGuard;

    private final LlmReasonValidator validator;

    private final LlmReasonMapper llmReasonMapper;

    private final RecommendMapper recommendMapper;

    private final RatingMapper ratingMapper;

    /** 冷启动勾选的影片（阶段六 §7.2）—— 新用户唯一的口味信号，见 {@link #loadHistorySafely} */
    private final UserPreferenceMapper userPreferenceMapper;

    /**
     * 启动时检查一处**容易误以为生效**的配置（REVIEW-03 补）。
     *
     * <p>{@code app.llm.rerank-enabled} 是 R16 保留的消融开关，但阶段四的在线路径
     * **没有**接重排（依据 EXP-03：采样口径下重排使 HR@10 由 0.6520 降到 0.4200）。
     * 于是它现在是个"改了也不会生效"的开关 —— 与其静默，不如启动时明说，
     * 免得有人在答辩前改了它、以为首页顺序会变。
     */
    @PostConstruct
    void warnIfRerankEnabled() {
        if (props.getLlm().isRerankEnabled()) {
            log.warn("app.llm.rerank-enabled=true，但阶段四的在线路径未接入重排（R16 默认关闭）："
                    + "该开关目前不改变任何接口行为。重排消融实验请用 "
                    + "script/llm_rerank_ablation.py（Python 侧，与论文口径一致）。");
        }
    }

    @Override
    public void attachReasons(long userId, RecommendVO vo) {
        try {
            doAttachReasons(userId, vo);
        } catch (Exception e) {
            // ★ 兜底：接口契约第一条是"不抛异常"。走到这里说明出现了没预料到的异常
            //   （例如数据库不可用）。此时连模板理由都可能挂不上，但**绝不能**因此
            //   让首页打不开 —— 推荐结果本身是好的，缺一句理由只是降级展示。
            //   这条由 LlmReasonServiceTest#unexpectedFailureNeverPropagates 守住。
            log.warn("推荐理由挂载失败（不影响推荐结果本身）：{}", e.toString());
        }
    }

    private void doAttachReasons(long userId, RecommendVO vo) {
        List<RecommendItemVO> items = (vo == null) ? null : vo.getItems();
        if (items == null || items.isEmpty()) {
            return;
        }
        int topN = Math.min(Math.max(props.getLlm().getReasonTopN(), 0), items.size());
        if (topN == 0) {
            return;
        }
        List<Long> targetIds = items.subList(0, topN).stream()
                .map(RecommendItemVO::getMovieId).toList();
        Map<Long, RecommendItemVO> byId = new LinkedHashMap<>();
        // 单条理由要挂回 VO 上的那个对象，所以这里必须持有引用而不是重建；
        // putIfAbsent 同时起到"推荐列表里出现重复影片时只保留第一条"的作用
        items.subList(0, topN).forEach(item -> byId.putIfAbsent(item.getMovieId(), item));
        List<MyRatingVO> history = loadHistorySafely(userId);

        LocalDate batchDate = resolveBatchDate(userId, vo.getSource());

        // 第 1 步：缓存（成本控制的第一条，设计文档 §5.1）
        //
        // ★ 缓存必须排在"大模型开关"之前（2026-09-25 调整）：原先 `enabled=false` 会**直接**填模板返回，
        //   于是即使库里已有真实的 AI 理由也一律不显示 —— 演示账号的"离线预生成"就白做了。
        //   现在顺序是：先看缓存 → 命中的照常用（这是已经生成过的真实输出，与模型此刻可不可用无关）；
        //   只有**仍缺**的那些才去看模型能不能调（开关/熔断/上限），不能调才补模板 ——
        //   且补的模板**不落库**，免得把模板写成缓存（缓存命中是不区分来源的）。
        Map<Long, LlmReason> cached = loadCacheSafely(userId, batchDate, targetIds);
        cached.values().forEach(row -> {
            RecommendItemVO item = byId.get(row.getMovieId());
            if (item != null) {
                applyCached(item, row);
            }
        });

        // distinct()：推荐列表理论上可能含重复影片，去重后既少一次落库、也不会
        // 给同一个对象挂两遍理由（唯一键是 (user_id, movie_id)，重复写只是白跑）。
        List<RecommendItemVO> missing = targetIds.stream()
                .distinct()
                .filter(id -> !cached.containsKey(id))
                .map(byId::get)
                .filter(Objects::nonNull)
                .toList();
        if (missing.isEmpty()) {
            log.debug("用户 {} 的 {} 条理由全部命中缓存（批次 {}）", userId, byId.size(), batchDate);
            return;
        }

        // ★ 协同依据（用户提问"协同过滤体现在哪"后补）：只对**要生成理由**的这几条去查。
        //   缓存命中的不查 —— 于是"已有缓存"这条路径（演示账号走的正是它）完全不发生额外查询。
        Map<Long, CollaborativeHint> hints = findCollaborativeHints(userId, missing, history);

        // 第 2 步：未命中的部分调一次大模型（一次调用产出全部理由，不是每条一次）。
        // 大模型整体关闭时**只就地补模板、不落库** —— 否则会把模板写成"缓存"，
        // 以后模型可用了也会命中这条模板（缓存命中不区分来源）。
        if (!props.getLlm().isEnabled()) {
            log.debug("大模型已关闭：{} 条未命中缓存的理由就地补模板（不落库）", missing.size());
            missing.forEach(item -> fillTemplate(item, history, hints.get(item.getMovieId())));
            return;
        }
        Map<Long, String> generated = Map.of();
        if (llmGuard.allowCall()) {
            generated = generateReasons(userId, history, items, byId, targetIds, hints);
        } else {
            log.debug("大模型闸门未放行（熔断/超当日上限），{} 条理由走模板降级", missing.size());
        }

        // 第 3 步：逐条落库并挂到 VO；模型没给出的用模板补齐（设计文档 §5.4 第 3 层）
        for (RecommendItemVO item : missing) {
            String reason = generated.get(item.getMovieId());
            boolean fromModel = StringUtils.hasText(reason);
            if (!fromModel) {
                reason = templateReason(item, history, hints.get(item.getMovieId()));
            }
            apply(item, reason, fromModel);
            save(userId, item.getMovieId(), reason, fromModel, batchDate);
        }
    }

    /**
     * 读"用户喜欢的影片"作为口味上下文。**失败返回已有的那部分，绝不抛异常。**
     *
     * <p>两个来源合并（阶段六 §7.2 的联动）：
     * <ol>
     *   <li>{@code rating} 的高分影片 —— 最强的口味信号；</li>
     *   <li>{@code user_preference} 里冷启动勾选的影片 —— 新用户**唯一**的信号。
     *       没有这一条，刚勾选完 5 部影片的用户拿到的仍然是一句通用的
     *       "近期热门：N 人评出 X 分"，与他亲手做的选择毫无关系 ——
     *       那正是冷启动"花了力气勾选却看不出区别"的典型症状。</li>
     * </ol>
     * 同一部影片两边都有时**以评分为准**（评分是更强、更新的表达），
     * 排序按分数降序取前 {@value #HISTORY_LIMIT} 条。
     *
     * <p>空历史不会让功能不可用，只会让模板文案从"因为你喜欢《XX》"退到事实型那套 ——
     * 那套本来就是给没有口味信号的用户准备的，所以这条降级路径是现成的。
     */
    private List<MyRatingVO> loadHistorySafely(long userId) {
        Map<Long, MyRatingVO> merged = new LinkedHashMap<>();
        // 先放偏好（弱信号），再放评分（强信号）—— 后者按 movieId 覆盖前者
        collectTaste(merged, () -> userPreferenceMapper.selectAsTasteContext(userId, HISTORY_LIMIT),
                "冷启动偏好");
        collectTaste(merged, () -> ratingMapper.selectTopRated(userId, HISTORY_LIMIT),
                "用户高分影片");
        return merged.values().stream()
                .sorted((a, b) -> Integer.compare(scoreOf(b), scoreOf(a)))
                .limit(HISTORY_LIMIT)
                .toList();
    }

    private void collectTaste(Map<Long, MyRatingVO> target,
                              java.util.function.Supplier<List<MyRatingVO>> loader,
                              String what) {
        try {
            List<MyRatingVO> rows = loader.get();
            if (rows == null) {
                return;
            }
            rows.forEach(row -> {
                if (row != null && row.getMovieId() != null) {
                    target.put(row.getMovieId(), row);
                }
            });
        } catch (Exception e) {
            log.warn("读{}失败，这部分口味信号本次忽略：{}", what, e.toString());
        }
    }

    private int scoreOf(MyRatingVO row) {
        return row.getScore() == null ? 0 : row.getScore();
    }

    /**
     * 为**每条待生成理由**的候选影片找一条"协同依据"。
     *
     * <h3>★ 为什么要做（使用者提问"协同过滤体现在哪，目前没有很明显的表现"）</h3>
     * 首页推荐来源本就是 NeuMF（协同过滤），但**理由讲的全是内容侧**（"经典悬疑，耐人寻味"）——
     * 于是用户与评委都看不出协同过滤在哪里起了作用。协同过滤最容易被理解的一句话恰恰是
     * "**和你口味相近的人也在看这部**"，而这句话我们有真实数据支撑：候选影片与
     * **用户自己高分过的影片**之间，有多少用户是同时喜欢的。
     *
     * <h3>怎么算（复用已有查询，不新增 SQL）</h3>
     * 对每部候选，取它与全库的共现邻居（{@code RecommendMapper#selectAlsoLiked}，
     * 已按共同喜欢人数倒序），再与"用户的高分影片"求交 —— 命中即得依据
     * "喜欢《A》的 N 位用户也很喜欢这部"。单次约 0.1~0.25 秒（见 TEST-12 的性能实测），
     * 5 条约 0.5~1 秒，藏在一次 8 秒的模型调用后面，用户感知不到；
     * 而**有缓存时根本不查**（见调用点）。
     *
     * <p>失败一律当作"没有依据"：协同依据是锦上添花，缺了理由照常生成。
     * {@code rating_high} 派生表还没建的库会直接抛异常 —— 这里**在第一次失败后就退出**，
     * 不打 5 条重复日志（那会让人以为系统坏了）。
     */
    private Map<Long, CollaborativeHint> findCollaborativeHints(long userId,
                                                               List<RecommendItemVO> missing,
                                                               List<MyRatingVO> history) {
        if (missing == null || missing.isEmpty() || history == null || history.isEmpty()) {
            return Map.of();
        }
        Set<Long> candidateIds = new LinkedHashSet<>();
        for (RecommendItemVO item : missing) {
            if (item != null && item.getMovieId() != null) {
                candidateIds.add(item.getMovieId());
            }
        }
        if (candidateIds.isEmpty()) {
            return Map.of();
        }

        // ★ 方向很关键：**从"用户喜欢的影片"出发找邻居**，而不是从候选出发。
        //   实测（2026-10-03，演示账号 4277）：反过来一条都找不到 ——
        //   用户的高分片往往冷门（《War Zone, The》这种），而热门候选的邻居榜被更热门的片占满，
        //   冷门锚点挤不进前 30 ✗。反过来，冷门片的邻居榜恰恰由"它那一小撮观众也喜欢的片"构成，
        //   候选就在里面 ✓（实测 3 个锚点命中 2 部候选）。
        //   → 教训：**共现查询的两个方向不等价**，要看数据长什么样再定。
        Map<Long, CollaborativeHint> hints = new LinkedHashMap<>();
        for (MyRatingVO anchor : history) {
            if (anchor == null || anchor.getMovieId() == null) {
                continue;
            }
            String anchorTitle = displayTitle(anchor);
            try {
                for (AlsoLikedVO partner : recommendMapper.selectAlsoLiked(anchor.getMovieId(),
                        HINT_NEIGHBOURS)) {
                    if (partner.getMovieId() == null || !candidateIds.contains(partner.getMovieId())) {
                        continue;
                    }
                    long together = partner.getTogether() == null ? 0L : partner.getTogether();
                    // 同一部候选可能被多个锚点命中 —— 取共同人数最多的那条（证据更强）
                    CollaborativeHint old = hints.get(partner.getMovieId());
                    if (old == null || together > old.together()) {
                        hints.put(partner.getMovieId(), new CollaborativeHint(anchorTitle, together));
                    }
                }
            } catch (Exception e) {
                log.warn("查协同依据失败，本次理由不带协同依据（不影响理由生成）：{}", e.toString());
                return hints;
            }
        }
        if (!hints.isEmpty()) {
            hints.forEach((movieId, hint) -> log.info(
                    "用户 {}：候选 {} 的协同依据 = 喜欢《{}》的 {} 位用户也很喜欢这部",
                    userId, movieId, hint.anchorTitle(), hint.together()));
        }
        log.info("用户 {}：{} 条待生成理由中，{} 条找到了协同依据", userId, candidateIds.size(), hints.size());
        return hints;
    }

    /** 协同依据的文本形式（给 Prompt 用；模板理由自己拼，因为要更短的措辞） */
    private static Map<Long, String> hintTexts(Map<Long, CollaborativeHint> hints) {
        if (hints == null || hints.isEmpty()) {
            return Map.of();
        }
        Map<Long, String> texts = new LinkedHashMap<>();
        hints.forEach((movieId, hint) -> texts.put(movieId,
                "喜欢《" + hint.anchorTitle() + "》的 " + hint.together() + " 位用户也喜欢这部"));
        return texts;
    }

    /** 读理由缓存；**失败按"全部未命中"处理**（最坏是重算一次，不影响可用性） */
    private Map<Long, LlmReason> loadCacheSafely(long userId, LocalDate batchDate, List<Long> targetIds) {
        try {
            return llmReasonMapper.selectByUserAndBatch(userId, batchDate, targetIds).stream()
                    .collect(Collectors.toMap(LlmReason::getMovieId, row -> row, (a, b) -> a));
        } catch (Exception e) {
            log.warn("读推荐理由缓存失败，本次按未命中处理（会重新生成一次）：{}", e.toString());
            return Map.of();
        }
    }

    /** 调一次模型；任何失败都返回空 Map（调用方据此走模板降级），不抛异常 */
    private Map<Long, String> generateReasons(long userId,
                                              List<MyRatingVO> history,
                                              List<RecommendItemVO> items,
                                              Map<Long, RecommendItemVO> byId,
                                              List<Long> requestedIds,
                                              Map<Long, CollaborativeHint> hints) {
        int contextSize = Math.min(
                Math.max(props.getLlm().getCandidateSize(), byId.size()), items.size());
        List<RecommendItemVO> context = List.copyOf(items.subList(0, contextSize));
        List<RecommendItemVO> requested = List.copyOf(byId.values());
        try {
            String raw = llmClient.completeJson(LlmPromptTemplates.REASON_SYSTEM,
                    LlmPromptTemplates.reasonUserPrompt(history, context, requested,
                            hintTexts(hints)));

            LlmReasonValidator.Result result = validator.validate(raw, requestedIds);
            if (result.parseFailed()) {
                // 整体不可解析说明"调用链条"没在正常工作，计入熔断计数。
                // ★ 这里**不能**先 recordSuccess()：那是"清零失败计数"，若在校验之前调用，
                //   一个持续返回垃圾的模型就会"成功—失败"交替、连续失败数永远回到 0，
                //   熔断永远不会触发 —— 而这正是最该熔断的情形（这条由
                //   LlmReasonServiceTest#unparsableOutputCountsAsFailure 守住）。
                llmGuard.recordFailure();
                log.warn("用户 {} 的理由输出整体不可解析，已计入熔断计数", userId);
                return Map.of();
            }
            llmGuard.recordSuccess();

            if (!result.usable()) {
                log.warn("用户 {} 的理由输出全部被校验剔除，本次全部走模板降级", userId);
                return Map.of();
            }
            log.debug("用户 {} 理由生成成功：可用 {} 条，剔除 {} 条", userId, result.reasons().size(), result.rejected());
            return result.reasons();
        } catch (LlmException e) {
            llmGuard.recordFailure();
            log.warn("用户 {} 的推荐理由生成失败，走模板降级：{}", userId, e.getMessage());
            return Map.of();
        }
    }

    /**
     * 理由所属批次：离线路径与 {@code rec_result} 的最新批次对齐，
     * 兜底路径（热度榜/最新上架）没有批次概念，取当天。
     *
     * <p>这样"重训后旧理由整体失效"（§5.3）才有明确对象：重训产出新批次后，
     * 旧批次的理由再也不会被命中。
     */
    private LocalDate resolveBatchDate(long userId, String source) {
        if (RecommendServiceImpl.SOURCE_OFFLINE.equals(source)) {
            try {
                LocalDate batch = recommendMapper.selectLatestBatchDate(userId);
                if (batch != null) {
                    return batch;
                }
            } catch (Exception e) {
                log.warn("查离线批次日期失败，本条理由按当天批次处理：{}", e.toString());
            }
        }
        return LocalDate.now();
    }

    private void applyCached(RecommendItemVO item, LlmReason row) {
        boolean model = row.getSource() != null && row.getSource() == LlmReason.SOURCE_MODEL;
        item.setReason(row.getReason());
        item.setReasonSource(model
                ? RecommendItemVO.REASON_SOURCE_MODEL : RecommendItemVO.REASON_SOURCE_TEMPLATE);
    }

    private void apply(RecommendItemVO item, String reason, boolean fromModel) {
        item.setReason(reason);
        item.setReasonSource(fromModel
                ? RecommendItemVO.REASON_SOURCE_MODEL : RecommendItemVO.REASON_SOURCE_TEMPLATE);
    }

    private void fillTemplate(RecommendItemVO item, List<MyRatingVO> history,
                              CollaborativeHint hint) {
        apply(item, templateReason(item, history, hint), false);
    }

    /**
     * 模板理由（设计文档 §5.4）。
     *
     * <p>★ 文档只给了一句"因为你喜欢《XX》，为你推荐《YY》"，但那句**对没有评分历史的
     * 新注册账号不成立**（XX 取谁？）—— 而新账号恰恰是阶段三完成标志里要演示的路径。
     * 因此补第二套：用"多少人在评、评了多少分"这类**与用户无关但真实**的事实说话。
     * 另外推荐链路**不过滤已评分影片**（阶段三已定口径），所以会出现"被推荐的正是
     * 用户最喜欢的作品"，那种情况下第一套文案会变成"因为你喜欢《X》，为你推荐《X》"，
     * 也改走第二套。
     */
    private String templateReason(RecommendItemVO item, List<MyRatingVO> history,
                                  CollaborativeHint hint) {
        String target = displayTitle(item);
        // ① 有协同依据就**优先用它**（用户提问"协同过滤体现在哪"后补）。
        //    这条与后面两条的区别在**证据类型**：另两条讲"题材/热度"，这条讲"和你口味相近的人"
        //    —— 而后者才是协同过滤的解释，数字来自真实的评分共现，不是编的。
        if (hint != null) {
            return "喜欢《" + hint.anchorTitle() + "》的 " + hint.together() + " 位用户也很喜欢这部";
        }
        String favorite = (history == null || history.isEmpty()) ? null : displayTitle(history.get(0));
        if (StringUtils.hasText(favorite) && !favorite.equals(target)) {
            return "因为你喜欢《" + favorite + "》，为你推荐《" + target + "》";
        }
        if (item.getRatingCount() != null && item.getRatingCount() > 0 && item.getRatingAvg() != null) {
            return "近期热门：" + item.getRatingCount() + " 人评出 " + item.getRatingAvg() + " 分";
        }
        return "新上架影片，暂无足够评分";
    }

    private void save(long userId, long movieId, String reason, boolean fromModel, LocalDate batchDate) {
        LlmReason row = new LlmReason();
        row.setUserId(userId);
        row.setMovieId(movieId);
        row.setReason(reason);
        row.setSource(fromModel ? LlmReason.SOURCE_MODEL : LlmReason.SOURCE_TEMPLATE);
        row.setBatchDate(batchDate);
        try {
            llmReasonMapper.upsert(row);
        } catch (Exception e) {
            // 落库失败不影响本次返回（理由已在 VO 里）；这一条下一批会被重新生成
            log.warn("推荐理由落库失败（user={} movie={}）：{}", userId, movieId, e.getMessage());
        }
    }

    /** 展示用片名：有中文名优先中文名，与前端口径一致 */
    private String displayTitle(RecommendItemVO item) {
        return StringUtils.hasText(item.getTitleCn()) ? item.getTitleCn() : item.getTitle();
    }

    private String displayTitle(MyRatingVO item) {
        return StringUtils.hasText(item.getTitleCn()) ? item.getTitleCn() : item.getTitle();
    }

    /**
     * 一条协同依据：**用户自己高分过的某部影片**，与待解释的候选影片**共同被多少人喜欢**。
     *
     * @param anchorTitle 用户喜欢的影片名（理由里要引用它，所以存名字而不是 id）
     * @param together    同时给两部片打过高分（≥4）的用户数 —— 真实的共现计数
     */
    private record CollaborativeHint(String anchorTitle, long together) {
    }
}
