// SDK 快速验证——起一个 API server，用 fetch 调用 SDK 类
import { RepairAgent } from './src/index.js';

const agent = new RepairAgent({ baseUrl: 'http://127.0.0.1:8000', timeoutMs: 120_000 });

const h = await agent.health();
console.log('health:', h.status, 'arms:', h.supports_arms);

const arms = await agent.listArms();
console.log('arms:', arms.map((a) => a.name));

const r = await agent.repair({
  arm: 'A3', // offline, zero LLM
  dialogue: {
    messages: [
      { role: 'system', content: '你是客服助手。' },
      { role: 'user', content: '你好。' },
      { role: 'assistant', content: '您好，请问是张三吗？' },
      { role: 'user', content: '对。' },
      { role: 'assistant', content: '哦对了您的借款今天到期了。' },
    ],
  },
});
console.log(`repair: status=${r.status} ok=${r.ok} issues=${r.issues.length}`);
