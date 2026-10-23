package com.movierec.service;

/**
 * 大模型调用入口（阶段四，设计文档 §5.1"统一入口"）。
 *
 * <p>只做一件事：把系统提示与用户提示发出去，把模型回复的**正文**拿回来。
 * 三条职责边界（都刻意的）：
 * <ul>
 *   <li><b>不解析业务语义</b> —— 白名单、去重、补齐属于输出校验，在
 *       {@code LlmReasonService} 那一层，因为只有那里知道候选集是什么；</li>
 *   <li><b>不做降级决策</b> —— 失败一律抛 {@link com.movierec.common.LlmException}，
 *       由调用方决定返回模板理由（设计文档 §5.4）；</li>
 *   <li><b>不管熔断与限额</b> —— 那是 {@link LlmGuard} 的事，调用前先问它 {@code allowCall()}。</li>
 * </ul>
 *
 * <p>协议：OpenAI 兼容的 {@code POST {base-url}/chat/completions}，
 * 与阶段 0 的数据脚本（{@code db/gen_intro_llm.py}）同一套约定，换厂商只改配置。
 * 当前实配为智谱 {@code glm-4-flash}。
 */
public interface LlmClient {

    /**
     * 以 JSON 模式调用一次，返回模型输出的 JSON 文本。
     *
     * <p>实现会带上 {@code response_format={"type":"json_object"}} —— **这是必须的**：
     * 实测（2026-09-24）不带它时模型会把 JSON 包在 ```json 代码块里，
     * 甚至只输出 1 条就当完成，直接解析必然失败。
     * 相应地，{@code userPrompt} 里必须出现 "JSON" 字样（厂商约定）。
     *
     * @param systemPrompt 角色与硬约束（不含具体数据）
     * @param userPrompt   具体数据与输出格式要求
     * @return 模型输出的 JSON 文本（未做业务校验）
     * @throws com.movierec.common.LlmException 超时、连接异常、4xx/5xx、响应非 JSON、正文为空
     */
    String completeJson(String systemPrompt, String userPrompt);
}
