package com.movierec.service;

import com.movierec.entity.Genre;
import com.movierec.vo.AiSearchItemVO;
import com.movierec.vo.AiSearchVO;
import com.movierec.vo.MyRatingVO;
import com.movierec.vo.RecommendItemVO;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Prompt 模板（阶段四起，设计文档 §5.2）—— **分开写，不硬凑成一个**。
 *
 * <p>为什么必须分开：三者的输入不同（候选+历史 / 候选+历史+重排要求 / 一句自然语言）、
 * 输出结构不同（理由数组 / 重排后的 Top-5 / 结构化查询条件）、
 * **默认状态也不同**（理由生成默认启用；重排默认关闭、仅用于复现 EXP-03；条件抽取属于阶段六）。
 * 合成一个模板的结果是：改一处的措辞会连带影响另外两处的行为，实验也就不可复现了。
 * 复用的是{@code LlmClient} 那层调用封装，不是模板本身 —— 文档特意澄清过这一点。
 *
 * <h3>★ 写 Prompt 时踩到的两条硬约束（都来自实测）</h3>
 * <ol>
 *   <li><b>正文里必须出现 "JSON" 字样</b>：智谱的 {@code response_format=json_object}
 *       模式有此要求，缺了会直接报参数错误。</li>
 *   <li><b>必须逐条列出要求生成的影片</b>：实测（2026-09-24）只笼统说"为候选写理由"时，
 *       模型有一次只返回 1 条就当完成（completion_tokens=34）。明确"5 部、一部都不能少"
 *       之后 4 次调用全部返回 5 条。</li>
 * </ol>
 * 这两条与 {@code LlmReasonValidator} 的白名单校验是配套的：Prompt 尽量让模型别出错，
 * 校验兜住它万一出了错。
 */
public final class LlmPromptTemplates {

    private LlmPromptTemplates() {
    }

    /** 模板一：理由生成【默认启用，R16】—— 只产解释，**不改变任何排序** */
    public static final String REASON_SYSTEM = """
            你是电影推荐系统的推荐理由生成器。你的任务只有一个：为给定的推荐影片各写一句简短的中文推荐理由。
            必须严格遵守：
            1. 只输出 JSON，格式为 {"reasons":[{"movieId":数字,"reason":"理由"}]}
            2. 只为指定的影片写理由，movieId 必须与给定的一致；不得新增、不得改动、不得遗漏
            3. 每条理由不超过 25 个汉字，一句话，要具体（可以是题材、看点、与用户口味的关联）
            4. 不要输出任何解释性文字，不要使用 Markdown 代码块
            5. 若某部影片带「协同依据」，它的理由**必须用上这个依据**：写出其中的片名与人数（如"喜欢《X》的 N 位用户也爱看"），措辞可微调但片名与数字必须一字不差；没带依据的影片才按题材与用户口味写
            """;

    /**
     * 模板二：重排 + 理由【可选，**默认关闭**，见 R16】。
     *
     * <p>保留它是为了能复现 EXP-03 的消融实验、支持论文复现；**线上不启用**
     * —— 依据 EXP-03 实测，采样候选池下重排使 HR@10 由 0.6520 降到 0.4200
     * （p&lt;1e-6，显著变差）。
     *
     * <p>★ 本阶段**没有**把它接到在线路径上：{@code app.llm.rerank-enabled=true} 不改变
     * 任何接口行为，只会在应用启动时打一条 WARN 提醒（见
     * {@code LlmReasonServiceImpl#warnIfRerankEnabled}），避免"改了配置却以为生效了"。
     * 真正的重排实验在 {@code script/llm_rerank_ablation.py}（Python 侧，与论文口径一致）。
     */
    public static final String RERANK_SYSTEM = """
            你是电影推荐系统的重排序与解释生成器。
            在给定候选影片的基础上，选出最适合该用户的 5 部并重新排序，同时为这 5 部各写一句中文推荐理由。
            必须严格遵守：
            1. 只输出 JSON，格式为 {"items":[{"movieId":数字,"reason":"理由"}]}，按你给出的推荐顺序排列
            2. movieId 只能取自给定候选，不得编造
            3. 每条理由不超过 25 个汉字
            4. 不要输出任何解释性文字，不要使用 Markdown 代码块
            """;

    /**
     * 模板三：条件抽取（自然语言 → 结构化条件），**阶段六「AI 找片」使用**。
     *
     * <p>阶段四只把模板写下来、不接线；2026-09-25 实现 AI 找片时接线，并按数据事实调整了两处：
     * <ul>
     *   <li><b>去掉 {@code director}</b>：库里 {@code director} 只有 10 条（0.3%，实测），
     *       抽出来也没法按它过滤，留着只会给出"条件对了却零结果"的假象；</li>
     *   <li><b>加上 {@code similarTo}</b>：用户常会说"像《盗梦空间》那样的" ——
     *       把片名抽出来，后端据此查该片的类型做相似检索（库里没有向量语义检索）。</li>
     * </ul>
     *
     * <p>安全约束（设计文档 §7.4，实现时逐条遵守）：模型只输出**结构化条件**；
     * SQL 一律参数化（{@code apply} 的 {0} 占位符）；排序/分页/字段由后端固定；
     * **禁止执行模型输出的任何 SQL 片段**。
     */
    public static final String FILTER_SYSTEM = """
            你是电影检索条件抽取器。把用户的一句自然语言需求，抽取成结构化查询条件。
            必须严格遵守：
            1. 只输出 JSON，格式为 {"genres":[字符串],"yearFrom":整数或null,"yearTo":整数或null,"keyword":字符串或null,"similarTo":字符串或null}
            2. genres 只能取自下面给出的类型清单，用清单里的中文名，最多 3 个；不确定就给空数组
            3. similarTo 只在用户明确提到某部影片时填该片名，否则填 null
            4. keyword 填用户描述里的题材/风格关键词（如"烧脑""治愈"），没有就给 null
            5. 无法确定的字段填 null，不要猜测、不要编造
            6. 不要输出任何解释性文字，不要输出 SQL
            """;

    /**
     * 理由生成模板的用户提示：用户口味（历史高分影片）+ 要解释的候选影片。
     *
     * @param history    用户评分最高的若干部影片，可为空（新注册账号）
     * @param candidates 喂给模型的候选上下文（含影片信息，帮助模型写出具体的理由）
     * @param requested  要求逐条给出理由的影片（**顺序即展示顺序**）
     */
    public static String reasonUserPrompt(List<MyRatingVO> history,
                                         List<RecommendItemVO> candidates,
                                         List<RecommendItemVO> requested) {
        return reasonUserPrompt(history, candidates, requested, Map.of());
    }

    /**
     * 理由生成的用户提示（**带协同依据版**，2026-10-03 补）。
     *
     * <p>★ 为什么要把"协同依据"喂给模型：首页推荐来源本就是 NeuMF（协同过滤），
     * 而原先的理由只讲题材与热度，用户与评委都看不出协同过滤起了什么作用。
     * 给每条候选附一句"喜欢《A》的 N 位用户也喜欢这部"（A 是**该用户自己高分过的影片**，
     * 数字来自真实评分共现），模型就能写出协同向的理由 —— 这是"看得见的协同过滤"。
     *
     * @param hints movieId → 协同依据文本；没有的影片不带（模型按题材写），可为空
     */
    public static String reasonUserPrompt(List<MyRatingVO> history,
                                         List<RecommendItemVO> candidates,
                                         List<RecommendItemVO> requested,
                                         Map<Long, String> hints) {
        StringBuilder sb = new StringBuilder();
        if (history == null || history.isEmpty()) {
            sb.append("该用户还没有评分记录（新用户），没有可用的口味历史。\n");
        } else {
            sb.append("该用户评分最高的影片：\n");
            for (MyRatingVO row : history) {
                sb.append("- ").append(row.getMovieId()).append(" 《").append(title(row))
                        .append("》(").append(row.getYear()).append(") 他给了 ")
                        .append(row.getScore()).append(" 分\n");
            }
        }

        sb.append("\n请为以下 ").append(requested.size())
                .append(" 部候选影片**各写一句**推荐理由，一部都不能少：\n");
        for (RecommendItemVO item : requested) {
            sb.append("- ").append(item.getMovieId()).append(" 《").append(title(item)).append("》(")
                    .append(item.getYear()).append(")");
            if (item.getRatingCount() != null && item.getRatingCount() > 0) {
                sb.append(" 均分 ").append(item.getRatingAvg())
                        .append("（").append(item.getRatingCount()).append(" 人评）");
            }
            String hint = (hints == null) ? null : hints.get(item.getMovieId());
            if (StringUtils.hasText(hint)) {
                sb.append("  ← 协同依据：").append(hint).append("（这条理由必须用上它）");
            }
            sb.append("\n");
        }

        if (candidates != null && candidates.size() > requested.size()) {
            sb.append("\n（参考：本次推荐池共 ").append(candidates.size())
                    .append(" 部影片，你只需要为上面列出的 ").append(requested.size())
                    .append(" 部写理由。）\n");
        }
        sb.append("\n只输出 JSON。");
        return sb.toString();
    }

    /**
     * 条件抽取模板的用户提示：用户原话 + **允许的类型清单**。
     *
     * <p>为什么要把清单喂给它：不给清单时模型会自由发挥（"科幻片" → "科幻" / "Sci-Fi" /
     * "science fiction" 三种写法都有），后端还得做模糊匹配；给了清单并要求"只能取自清单"，
     * 绝大多数情况直接命中。这与"输出必须过白名单校验"是配套的：
     * Prompt 尽量让它别错，校验兜住它万一出错。
     */
    public static String filterUserPrompt(String query, List<Genre> genres) {
        return filterUserPrompt(query, genres, List.of());
    }

    /**
     * 条件抽取的用户提示（**多轮版**）：原话 + 类型清单 + 之前几轮说过的话。
     *
     * <p>★ 多轮的关键在这句指令上：用户追问时说的往往是**补充**（"科幻片" → "90 年代的"），
     * 而不是改口。如果只把当前这句抽成条件，检索范围会**变大**（从"科幻 + 90 年代"退化成
     * "90 年代不管什么类型"）—— 与用户的直觉正好相反。所以明确要求它输出**合并后**的完整条件，
     * 这样"再加一个条件，结果变少"才成立。
     *
     * <p>同时把历史标注成"越靠后越新"，让模型在**真改口**（"不要爱情了，改看恐怖"）时
     * 知道以哪句为准。
     */
    public static String filterUserPrompt(String query, List<Genre> genres, List<String> history) {
        String allowed = genres.stream()
                .map(g -> StringUtils.hasText(g.getNameCn()) ? g.getNameCn() : g.getName())
                .collect(Collectors.joining("、"));
        StringBuilder sb = new StringBuilder();
        sb.append("可用的类型清单（genres 只能从这些里面选，用中文名）：").append(allowed).append("\n\n");

        if (history != null && !history.isEmpty()) {
            sb.append("用户在前几轮还说过（越靠后越新）：\n");
            for (int i = 0; i < history.size(); i++) {
                sb.append(i + 1).append(". ").append(history.get(i)).append("\n");
            }
            sb.append("\n注意：上面那些话只用来帮你**理解省略了主语的追问**"
                    + "（例如「那 90 年代的呢」指的是什么）。\n")
                    .append("★ 但条件**只填当前这一句明确表达的**：不要抄上文的条件，"
                            + "也不要替用户补充任何类型或年份 —— 多轮条件的合并由后端按确定性规则完成，"
                            + "你多填一个就会把结果搞错。\n\n");
        }

        sb.append("用户这一轮的需求：").append(query).append("\n\n只输出 JSON。");
        return sb.toString();
    }

    /**
     * 模板五：AI 找片的"回答 + 逐条说明"（**一次调用产出两者**）。
     *
     * <p>★ 为什么与模板一/三/四分开写：2026-09-25 用户反馈"大模型痕迹不足" ——
     * AI 找片的回复原来只有前端写死的一句"找到 30 部符合的影片"，**看不到模型说话**。
     * 补"自然语言回答"有两条路：再发一次调用（多等 5~8 秒），或者**与逐条说明合并成一次**。
     * 选后者：这两件事本来就是同一次"编辑视角"的活儿（先给整体推荐语，再逐条说为什么合适），
     * 合并后**调用次数与总耗时都不变**。
     *
     * <p>为什么不是模板一加个字段：模板一被推荐理由路径（{@code LlmReasonServiceImpl}）共用，
     * 给它加 {@code reply} 会让那条路径平白多出一个无用字段、也让"改了措辞影响谁"变得不清楚
     * —— 这正是本类开头强调"分开写"的原因。**产物形状变了，就该另开模板**。
     */
    public static final String MATCH_SYSTEM = """
            你是电影推荐编辑。用户说了一句想找什么电影，系统已经检索出候选影片。
            你要做两件事：写一段推荐语，并为指定的影片各写一句"为什么符合"。
            必须严格遵守：
            1. 只输出 JSON，格式为 {"reply":"推荐语","reasons":[{"movieId":数字,"reason":"理由"}]}
            2. reply 是 3~5 句连贯的中文，口语自然，像编辑跟读者说话；不要分条、不要换行、不要 Markdown、不要任何前缀（如"好的"）
            3. reply 里先回应"听懂了什么需求"，再从候选里挑 2~3 部说明为什么值得看；提到影片必须写《片名》
            4. reply 里**只能提到候选列表里的影片**，不得提到列表之外的任何影片
            5. reasons 只为指定影片写，movieId 必须与给定的一致；不得新增、不得改动、不得遗漏；每条不超过 25 个汉字
            6. 不得编造候选列表之外的信息（演员、情节、评分一律以给定为准）
            7. 不要输出任何解释性文字，不要使用 Markdown 代码块
            """;

    /**
     * "回答 + 逐条说明"的用户提示：用户原话 + **已校验的条件** + 候选影片。
     *
     * <p>为什么要把"已校验的条件"喂进去：回答里能自然带一句"按你说的科幻 + 90 年代筛的"，
     * 这正是让用户看见"模型听懂了"的地方；同时它知道的也只有这些条件，不会自由发挥。
     */
    public static String matchUserPrompt(String query, AiSearchVO.Understood understood,
                                         List<AiSearchItemVO> items) {
        StringBuilder sb = new StringBuilder();
        sb.append("用户的原话：").append(query).append("\n");

        List<String> parts = new ArrayList<>();
        if (understood != null && !understood.getGenres().isEmpty()) {
            parts.add("类型=" + String.join("/", understood.getGenres()));
        }
        if (understood != null && understood.getYearFrom() != null && understood.getYearTo() != null) {
            parts.add("年份=" + understood.getYearFrom() + "~" + understood.getYearTo());
        } else if (understood != null && understood.getYearFrom() != null) {
            parts.add("年份=" + understood.getYearFrom() + " 年之后");
        } else if (understood != null && understood.getYearTo() != null) {
            parts.add("年份=" + understood.getYearTo() + " 年之前");
        }
        if (understood != null && StringUtils.hasText(understood.getKeyword())) {
            parts.add("风格关键词=" + understood.getKeyword());
        }
        if (understood != null && StringUtils.hasText(understood.getSimilarTo())) {
            parts.add("类似《" + understood.getSimilarTo() + "》");
        }
        sb.append("系统理解到的条件（已校验，推荐语里可以直接用这些说法）：")
                .append(parts.isEmpty() ? "（没解析出明确条件，是按关键词找的）" : String.join("，", parts))
                .append("\n");

        sb.append("\n候选影片（共 ").append(items.size()).append(" 部，按评分人数排序）：\n");
        for (AiSearchItemVO item : items) {
            sb.append("- ").append(item.getMovieId()).append(" 《").append(title(item)).append("》(")
                    .append(item.getYear()).append(")");
            if (item.getRatingCount() != null && item.getRatingCount() > 0) {
                sb.append(" 均分 ").append(item.getRatingAvg())
                        .append("（").append(item.getRatingCount()).append(" 人评）");
            }
            sb.append("\n");
        }
        sb.append("\n请为以上 ").append(items.size())
                .append(" 部影片**各写一句**为什么符合，一部都不能少，并给出整体推荐语。只输出 JSON。");
        return sb.toString();
    }

    /** 展示用片名（AI 找片结果） */
    private static String title(AiSearchItemVO item) {
        return StringUtils.hasText(item.getTitleCn()) ? item.getTitleCn() : item.getTitle();
    }

    /** 展示用片名：有中文名优先用中文名（与前端展示口径一致） */
    private static String title(RecommendItemVO item) {
        return StringUtils.hasText(item.getTitleCn()) ? item.getTitleCn() : item.getTitle();
    }

    private static String title(MyRatingVO item) {
        return StringUtils.hasText(item.getTitleCn()) ? item.getTitleCn() : item.getTitle();
    }
}
