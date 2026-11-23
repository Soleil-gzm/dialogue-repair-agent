/**
 * Dialogue Repair Agent SDK — TypeScript REST client.
 *
 * Usage:
 *   import { RepairAgent, DialogueMessage } from 'dialogue-repair-sdk';
 *
 *   const agent = new RepairAgent({ baseUrl: 'http://127.0.0.1:8000' });
 *   const result = await agent.repair({
 *     arm: 'A4',
 *     ignoreTurns: [2, 4],
 *     dialogue: {
 *       messages: [
 *         { role: 'system', content: '你是客服助手。' },
 *         { role: 'user', content: '【通话开始】' },
 *         { role: 'assistant', content: '您好，请问是张三吗？' },
 *         { role: 'user', content: '对。' },
 *         { role: 'assistant', content: '哦对了您的借款今天到期了。' },
 *         { role: 'user', content: '什么借款？' },
 *         { role: 'assistant', content: '您在小赢卡贷的借款，今天是还款截止日。' },
 *       ],
 *     },
 *   });
 *   console.log(result.status, result.ok, result.issues);
 */

export type Role = 'system' | 'user' | 'assistant';

export interface DialogueMessage {
  role: Role;
  content: string;
}

export interface Dialogue {
  messages: DialogueMessage[];
  [key: string]: unknown;
}

export interface RepairRequest {
  dialogue: Dialogue;
  arm?: 'A2' | 'A3' | 'A3L' | 'A3R' | 'A4' | 'A4R';
  reference?: Dialogue;
  ignoreTurns?: number[];
  runId?: string;
}

export interface Issue {
  rule?: string;
  turn?: number;
  detail?: string;
  [key: string]: unknown;
}

export interface RepairResponse {
  run_id: string;
  arm: string;
  status: string;
  termination: string;
  reason: string;
  iterations: number;
  tool_calls: number;
  tokens: number;
  ok: boolean;
  issues: Issue[];
  dialogue: Dialogue;
}

export interface ArmInfo {
  name: string;
  description: string;
}

export interface HealthInfo {
  status: string;
  agent_version: string;
  judge_version: string;
  rubric_version: string;
  supports_arms: string[];
  llm: {
    provider: string;
    model: string;
    base_url: string;
    reachable: boolean;
    [key: string]: unknown;
  };
}

export interface ClientOptions {
  baseUrl?: string;
  timeoutMs?: number;
}

export class RepairAgent {
  private baseUrl: string;
  private timeoutMs: number;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? 'http://127.0.0.1:8000').replace(/\/+$/, '');
    this.timeoutMs = options.timeoutMs ?? 60_000;
  }

  async health(): Promise<HealthInfo> {
    return this._get<HealthInfo>('/health');
  }

  async listArms(): Promise<ArmInfo[]> {
    return this._get<ArmInfo[]>('/api/v1/arms');
  }

  async repair(req: RepairRequest): Promise<RepairResponse> {
    const body = {
      dialogue: req.dialogue,
      arm: req.arm ?? 'A4',
      reference: req.reference,
      ignore_turns: req.ignoreTurns,
      run_id: req.runId,
    };
    // null fields omitted to keep payload clean
    const cleaned = Object.fromEntries(
      Object.entries(body).filter(([, v]) => v !== undefined)
    );
    return this._post<RepairResponse>('/api/v1/repair', cleaned);
  }

  private async _get<T>(path: string): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(this.timeoutMs) });
    if (!res.ok) throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
    return (await res.json()) as T;
  }

  private async _post<T>(path: string, body: unknown): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`POST ${url} failed: ${res.status} ${res.statusText}\n${text}`);
    }
    return (await res.json()) as T;
  }
}
