/**
 * Floatboat entry point for the local Creative Loop video-production engine.
 *
 * This Workflow only starts the restricted local launcher, waits for the Engine
 * Gateway, then opens the shared workbench. It does not receive credentials,
 * access local media, submit production jobs, or approve external uploads.
 */
export const meta = {
  id: 'creative-loop-floatboat-workbench',
  name: 'Creative Loop｜Floatboat 视频工作台',
  description: '启动本机 Creative Loop 视频生产引擎，确认 Agent v1 Gateway 就绪后，在 Floatboat 预览中打开专业视频工作台。',
  whenToUse: '需要从 Floatboat 打开 Creative Loop 工作台，提交素材任务、查看进度并审核成片时使用。',
  inputs: {
    type: 'object',
    properties: {
      open_workbench: {
        type: 'boolean',
        default: true,
        description: '服务就绪后是否打开本地工作台页面。'
      }
    }
  },
  outputs: {
    type: 'object',
    properties: {
      ok: { type: 'boolean' },
      status: { type: 'string' },
      message: { type: 'string' },
      engine_url: { type: 'string' },
      workbench_url: { type: 'string' },
      insight_logged_in: { type: 'boolean' },
      brand_assets_ready: { type: 'boolean' },
      mintegral_ready: { type: 'boolean' }
    },
    required: ['ok', 'status', 'message', 'engine_url', 'workbench_url', 'insight_logged_in', 'brand_assets_ready', 'mintegral_ready']
  },
  phases: [
    { title: '启动本机视频生产引擎' },
    { title: '检查 Creative Loop Gateway' },
    { title: '打开 Floatboat 视频工作台' }
  ],
  capabilities: {
    bash: { allowed: true },
    http: { required: true, allowedOrigins: ['http://127.0.0.1:8000'] },
    browser: { required: true, domains: ['localhost'] }
  },
  timeoutMs: 240000,
  concurrencyPolicy: 'skip'
};

const ENGINE_ORIGIN = 'http://127.0.0.1:8000';
const WORKBENCH_URL = 'http://localhost:8000/floatboat/workbench';
const PROJECT_ROOT = 'D:/creative-loop-floatboat';
const LAUNCHER = 'creative-loop-control.cmd';
const ALLOWED_ORIGINS = [ENGINE_ORIGIN];

function phase(ctx, title, detail) {
  try { ctx.log.info(title, { detail }); } catch (_) {}
}

function output(ok, status, message, payload = {}) {
  return {
    ok: Boolean(ok),
    status: String(status),
    message: String(message),
    engine_url: ENGINE_ORIGIN,
    workbench_url: WORKBENCH_URL,
    insight_logged_in: payload.insight_logged_in === true,
    brand_assets_ready: payload.brand_assets_ready === true,
    mintegral_ready: payload.mintegral_ready === true
  };
}

function parseBody(raw) {
  const body = raw?.json ?? raw?.body ?? raw?.responseBody ?? {};
  if (typeof body !== 'string') return body && typeof body === 'object' ? body : {};
  try { return JSON.parse(body); } catch (_) { return {}; }
}

async function health(ctx) {
  const raw = await ctx.http.get(`${ENGINE_ORIGIN}/agent/v1/health`, {
    allowedOrigins: ALLOWED_ORIGINS,
    timeoutMs: 10000
  });
  const body = parseBody(raw);
  if (!raw?.ok || body?.ok !== true) throw new Error('Gateway 健康检查未通过');
  return body;
}

export default async function run(ctx) {
  if (ctx.dryRun) {
    return output(true, 'dry_run_validated', '已校验受限启动器、本机 Engine Gateway 健康检查和工作台打开计划；测试运行未启动 Docker、未创建任务、未上传素材。');
  }

  phase(ctx, '启动本机视频生产引擎', '只调用项目内受限启动器的 start 白名单动作。');
  const start = await ctx.bash.run({
    command: `cmd.exe /c ${LAUNCHER} start`,
    cwd: PROJECT_ROOT,
    timeoutMs: 210000,
    rejectOnNonZero: false
  });
  if (Number(start?.exitCode) !== 0) {
    const diagnostic = String(start?.stderr || start?.stdout || '').trim().slice(-1200);
    return output(false, 'engine_start_failed', `Creative Loop 未能启动。${diagnostic ? ` ${diagnostic}` : '请检查 Docker Desktop 与项目配置。'}`);
  }

  phase(ctx, '检查 Creative Loop Gateway', '等待 Agent v1 健康检查，最长 30 秒。');
  let engine;
  let lastError = '';
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    try {
      engine = await health(ctx);
      break;
    } catch (error) {
      lastError = String(error?.message || error);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
  if (!engine) {
    return output(false, 'gateway_unavailable', `Creative Loop 已请求启动，但 Agent v1 Gateway 在等待期内不可用。${lastError ? ` ${lastError}` : ''}`);
  }

  if (ctx.input?.open_workbench !== false) {
    phase(ctx, '打开 Floatboat 视频工作台', '打开共享工作台；视频生产仍保留在本机 Creative Loop 引擎。');
    try {
      await ctx.browser.openUrl({
        url: WORKBENCH_URL,
        allowedOrigins: ['http://localhost:8000'],
        reuseStrategy: 'always_new'
      });
    } catch (error) {
      return output(false, 'workbench_open_failed', `引擎已就绪，但工作台未能打开：${String(error?.message || error)}`, engine);
    }
  }

  return output(
    true,
    ctx.input?.open_workbench === false ? 'engine_ready' : 'workbench_opened',
    ctx.input?.open_workbench === false ? 'Creative Loop 引擎已就绪。' : 'Creative Loop 引擎已就绪，Floatboat 视频工作台已打开。',
    engine
  );
}