package com.movierec.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * 本项目自有配置（前缀 {@code app}），与 spring / mybatis-plus 的配置分开，避免混淆。
 */
@Data
@Component
@ConfigurationProperties(prefix = "app")
public class AppProperties {

    private Admin admin = new Admin();

    /** 默认管理员账号：仅用于"库里连一个管理员都没有"时的一次性初始化。 */
    @Data
    public static class Admin {

        private String username = "admin";

        /** 口令不设默认值，必须由 app.admin.password 从外部配置注入。 */
        private String password = "";

        private String nickname = "系统管理员";
    }

    private Llm llm = new Llm();

    /**
     * 阶段四：大模型（推荐理由）相关配置。
     *
     * <p>接口走 OpenAI 兼容的 {@code /chat/completions} 协议，与阶段 0 的数据脚本
     * （{@code db/gen_intro_llm.py} 等）是同一套约定：换厂商只改 base-url / api-key / model
     * 三个值（智谱 / DeepSeek / 通义 / 硅基流动 / 本地 Ollama 都支持）。
     *
     * <p>★ 三项必填值（base-url / api-key / model）只放 {@code application-local.yml}
     * （已被 .gitignore 忽略），与数据库密码同一处 —— 这样 IDE 里跑与命令行跑 jar
     * 读的是同一份配置，**不需要任何环境变量**（见 TEST-04 的结论）。
     */
    @Data
    public static class Llm {

        /**
         * 总开关。关掉后推荐接口完全不调大模型，一律走模板理由。
         *
         * <p>用途：单元测试与 42 项 HTTP 验收脚本走这条 —— 否则验收会依赖外网与调用配额，
         * 网络一抖就把"验收失败"报成本项目的缺陷（那是两回事）。
         */
        private boolean enabled = true;

        private String baseUrl = "";

        private String apiKey = "";

        private String model = "";

        private double temperature = 0.6;

        private int maxTokens = 800;

        /**
         * 单次调用超时（毫秒）。
         *
         * <p>★ 实测依据（2026-09-24，glm-4-flash，要求输出 5 条理由的 JSON，共 4 次）：
         * 耗时 6.49 / 8.14 / 5.02 / 6.71 秒，均值 6.6 秒。
         * 设计文档 §5.1 原写"超时 5 秒"—— 按此实现会让**每一次在线调用都超时降级**，
         * 功能等于没接。故按实测改为 12 秒（约 1.5 倍实测上界），并保留为可调项。
         */
        private int timeoutMs = 12000;

        /**
         * 最多尝试几次（1 = 不重试）。
         *
         * <p>设计文档要求"仅对超时与 5xx 重试 1 次、4xx 不重试（重试也没用）"，
         * 故默认 2。最坏耗时 = 2 × timeout-ms；实测正常路径约 6.6 秒，重试是小概率分支。
         */
        private int maxAttempts = 2;

        /**
         * 每次调用要求模型生成几条理由（只给返回列表的前 N 条生成）。
         *
         * <p>见设计文档 §5.2 的口径修订：原文"候选 20 部 → 每部一句理由"与其余五处
         * （§5.1 校验 Top-5、§5.4 降级返 Top-5、§6.1 首页 Top-5 卡、§7.3 预生成 5 条）
         * 不一致，本次统一为 5 —— 前端只展示 5 张卡，多生成的用户看不到。
         * 第 6~N 条的 reason 为 null，这是约定而非缺陷。
         */
        private int reasonTopN = 5;

        /** 喂给模型的候选条数（接口默认返回 20 条，作为"为什么是这几部"的上下文） */
        private int candidateSize = 20;

        /**
         * 重排开关：**默认关闭**（设计文档 R16）。
         *
         * <p>依据 EXP-03 消融实测：采样候选池下 LLM 重排使 HR@10 由 0.6520 降到 0.4200
         * （p&lt;1e-6，显著变差）。线上直接采用召回顺序，LLM 只补理由；
         * 开启此开关走"重排 + 理由"模板，仅用于复现实验。
         */
        private boolean rerankEnabled = false;

        /** 每日调用上限；超限后直接走模板降级（成本控制，设计文档 §5.1） */
        private int dailyCallLimit = 2000;

        /** 熔断：连续失败达到多少次后进入熔断 */
        private int circuitFailureThreshold = 5;

        /** 熔断后的冷却时长（毫秒），冷却期内直接降级、不再发起调用 */
        private long circuitCooldownMs = 120000;
    }

    private Recommend recommend = new Recommend();

    /**
     * 推荐链路（2026-10-13 新增）。
     *
     * <p>★ 与 {@link Llm#isRerankEnabled()} 是**两件不同的事**，命名上刻意分开：
     * <ul>
     *   <li>{@code app.llm.rerank-enabled} = 让**大模型**重排名次 —— EXP-03 消融实测变差
     *       （HR@10 0.6520 → 0.4200，p&lt;1e-6），已如实关闭，只在复现实验时打开；</li>
     *   <li>{@code app.recommend.taste-rerank} = 用**用户自己的真实交互信号**（最近评分的口味）
     *       对离线召回做轻量重排 —— 本类下面这一组。两者机制完全不同，
     *       名字混用会让论文与答辩口径打架（"你们不是验证过重排无效吗？"）。</li>
     * </ul>
     */
    @Data
    public static class Recommend {

        private TasteRerank tasteRerank = new TasteRerank();

        /**
         * 实时口味重排（taste rerank）。
         *
         * <h3>★ 为什么默认关闭（2026-10-13 的决定）</h3>
         * 现有口径 R16 写明"线上顺序直接取召回结果"，而这条口径是**用实验立的**
         * （EXP-03：LLM 重排显著变差）。要改排序环节，就得用同一套标准 ——
         * 所以本轮的做法是：**先把能力实现出来、默认关，跑完对照实验再决定默认值**
         * （消融见 {@code script/taste_rerank_ablation.py} 与 EXP-04）。
         * 在拿到证据之前不动线上默认值，这才与项目一贯的处理方式一致。
         */
        @Data
        public static class TasteRerank {

            /**
             * 总开关。
             *
             * <h3>★ 为什么现在是 true（2026-10-14 的两次消融）</h3>
             * 第一轮（EXP-04 前半）：纯口味重排能把口味匹配率提上去（w=0.25 时 +10.5%），
             * 但**类目多样性掉 13%**、且前 10 平均只变 0.9 条 → 结论是"证据不支持默认开" ✓。
             * 第二轮（EXP-04 的 MMR 组）：加上多样性约束（{@link #diversityLambda}）后，
             * **w=0.25 / λ=0.30 做到了"口味 +2.2pp 且多样性高于基线"** ✓，
             * 代价是模型分落幅 **−1.87%**（超出我们预先声明的 1% 红线 ✗）。
             *
             * <p>★ **如实说明**：按事先声明的判定标准，这一组**不算达标** ——
             * 改默认值是**产品侧的决定**（"用 1.9% 的模型分落幅换口味与多样性"），
             * 不是实验结论。要回滚只需把这里改回 false（一个开关，没有别的耦合）。
             */
            private boolean enabled = true;

            /**
             * MMR 的多样性惩罚 λ：每轮选片时，从相关性里减掉
             * {@code λ × 与已选项的最大类型重合度（Jaccard）}。
             *
             * <p>0 = 不做多样性约束（就是第一轮消融里那个"多样性掉 13%"的版本）；
             * 0.30 = 第二轮消融（294 位用户 × 12 个组合）里**唯一**让"多样性回到基线之上"的档位。
             */
            private double diversityLambda = 0.30;

            /** 离线模型分（NeuMF 预测分）在最终打分里的权重 —— 必须最大，否则"来源：协同过滤"不成立 */
            private double weightModel = 0.60;

            /** 最近评分口味的权重 */
            private double weightTaste = 0.25;

            /** 新鲜度（年份）权重 */
            private double weightFresh = 0.10;

            /** 协同共现权重（本期未实现，预留给二期：单次查询 0.1~0.25 秒，会影响"首页秒开"） */
            private double weightCooccur = 0.00;

            /** 口味信号取"最近多少条评分" */
            private int tasteRecent = 30;

            /** 视为"喜欢"的最低分 */
            private int tasteMinScore = 4;

            /**
             * 至少要有多少条"喜欢的评分"才允许重排。
             *
             * <p>信号太薄时重排会**过拟合到一两部片**（比如刚评了一部科幻，整页就变成科幻）——
             * 那不是个性化，是抖动。少于这个条数只做"过滤已评分"，不动顺序。
             */
            private int minRatings = 3;
        }
    }

    private Poster poster = new Poster();

    /**
     * 阶段五：海报代理（让豆瓣图床的图从后端过一道）。
     *
     * <p>★ 为什么必须有它（2026-09-24 实测，不是推测）：
     * 豆瓣图床 {@code img*.doubanio.com} 开了**防盗链** —— 请求必须带
     * {@code Referer: https://movie.douban.com/} 才返回 200，否则**一律 418**。
     * 浏览器从 {@code http://localhost:5173} 发请求**带不上**豆瓣的 Referer，
     * 所以前端直接引用 poster_url 必然全部失败（实测 30+ 张图全 418，
     * 页面只能显示"暂无海报"占位图 —— 而库里其实有 893 张图可用）。
     * 后端能带这个头，于是改成"前端向后端要图、后端去取"。
     */
    @Data
    public static class Poster {

        /**
         * 本地缓存目录。留空则用 {@code 用户目录/.movie-rec/poster-cache}。
         *
         * <p>为什么默认放用户目录，而不是仓库内的 target：
         * <ol>
         *   <li>与进程工作目录无关 —— 命令行跑 jar（工作目录是仓库根）与 IDE 里跑
         *       （工作目录是 backend/）会解析到**同一个**目录，否则演示时会"预热到一个目录、
         *       读取另一个目录"，白白没有缓存；</li>
         *   <li>{@code mvn clean} 不会把它删掉（放 target 下就会，演示前被清空很难受）；</li>
         *   <li>不污染仓库，也不必新增 .gitignore 规则。</li>
         * </ol>
         *
         * <p>为什么不落盘不行：答辩现场可能没网/被墙。演示前跑一次
         * {@code python db/cache_posters.py} 预热，现场就完全不依赖外网。
         */
        private String cacheDir = "";

        /** 单张图的下载超时（毫秒）。取图失败只影响这一张（走占位图），不该拖慢页面。 */
        private int timeoutMs = 4000;

        /** 单张图的大小上限；超过就当作异常响应丢弃，避免把非图片内容当图存下来 */
        private int maxBytes = 4 * 1024 * 1024;
    }
}
