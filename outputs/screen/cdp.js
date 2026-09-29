#!/usr/bin/env node
'use strict';
// 零依赖 CDP 客户端（Node >= 21 自带 fetch / WebSocket）
// 用途：接管本机以 --remote-debugging-port=9223 启动的 Edge/Chrome，
//       用于人工登录态下的网页操作（如 AutoDL 控制台开机）。
// 用法：
//   node cdp.js tabs                列出所有目标
//   node cdp.js info                浏览器版本信息
//   node cdp.js nav <url>           导航并回读 title/url
//   node cdp.js text                读取当前页正文（前 6000 字）
//   node cdp.js html                读取当前页 outerHTML
//   node cdp.js eval "<js>"         执行 JS 并回读结果
//   node cdp.js shot <path.png>     截图当前页
const fs = require('fs');

const PORT = process.env.CDP_PORT || 9223;
const BASE = `http://127.0.0.1:${PORT}`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function listTargets() {
  let r;
  try {
    r = await fetch(`${BASE}/json/list`);
  } catch (e) {
    throw new Error(`连不上 CDP ${BASE} —— 浏览器是否以 --remote-debugging-port=${PORT} 启动？`);
  }
  if (!r.ok) throw new Error(`CDP 列表失败 HTTP ${r.status}`);
  return r.json();
}

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const pending = new Map();
    let seq = 0;
    const timer = setTimeout(() => reject(new Error('连接 CDP WebSocket 超时')), 10000);
    ws.addEventListener('open', () => {
      clearTimeout(timer);
      resolve({
        call(method, params = {}, timeoutMs = 30000) {
          const id = ++seq;
          return new Promise((res, rej) => {
            const t = setTimeout(() => {
              if (pending.has(id)) { pending.delete(id); rej(new Error(`命令超时: ${method}`)); }
            }, timeoutMs);
            pending.set(id, { res, rej, t });
            ws.send(JSON.stringify({ id, method, params }));
          });
        },
        close() { try { ws.close(); } catch (_) {} },
      });
    });
    ws.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (_) { return; }
      const p = pending.get(msg.id);
      if (!p) return;
      pending.delete(msg.id);
      clearTimeout(p.t);
      if (msg.error) p.rej(new Error(JSON.stringify(msg.error)));
      else p.res(msg.result);
    });
    ws.addEventListener('error', (e) => {
      clearTimeout(timer);
      reject(new Error('WebSocket 错误: ' + (e && (e.message || e.type))));
    });
  });
}

// 优先挑一个 http(s) 页面；新标签页(about:blank 等)排后面
async function pickPage() {
  const targets = await listTargets();
  const pages = targets.filter((t) => t.type === 'page');
  if (!pages.length) throw new Error('没有 page 类型目标（浏览器可能没开窗口）');
  const real = pages.find((p) => /^https?:/.test(p.url || ''));
  return real || pages[0];
}

async function withPage(fn) {
  const page = await pickPage();
  const c = await connect(page.webSocketDebuggerUrl);
  try { return await fn(c, page); } finally { c.close(); }
}

async function main() {
  const [cmd, ...args] = process.argv.slice(2);
  if (!cmd || cmd === 'help') {
    console.log('用法: node cdp.js <tabs|info|nav <url>|text|html|eval <js>|front|shot <path>>');
    process.exit(0);
  }

  if (cmd === 'tabs') {
    const ts = await listTargets();
    for (const t of ts) {
      console.log(`${t.type}\t${(t.title || '').slice(0, 50)}\t${(t.url || '').slice(0, 100)}`);
    }
    return;
  }

  if (cmd === 'info') {
    const r = await fetch(`${BASE}/json/version`);
    console.log(JSON.stringify(await r.json(), null, 2));
    return;
  }

  await withPage(async (c) => {
    if (cmd === 'nav') {
      if (!args[0]) throw new Error('nav 需要 url');
      await c.call('Page.enable');
      await c.call('Page.navigate', { url: args[0] });
      await sleep(3000);
      const r = await c.call('Runtime.evaluate', {
        expression: '({title: document.title, url: location.href})',
        returnByValue: true,
      });
      console.log(JSON.stringify(r.result.value));
    } else if (cmd === 'eval') {
      const r = await c.call('Runtime.evaluate', {
        expression: args.join(' '),
        returnByValue: true,
        awaitPromise: true,
      });
      const v = r.result && r.result.value;
      console.log(typeof v === 'string' ? v : JSON.stringify(v, null, 2));
    } else if (cmd === 'text') {
      const r = await c.call('Runtime.evaluate', {
        expression: '(document.body ? document.body.innerText : "").slice(0, 6000)',
        returnByValue: true,
      });
      console.log(r.result.value);
    } else if (cmd === 'html') {
      const r = await c.call('Runtime.evaluate', {
        expression: 'document.documentElement.outerHTML',
        returnByValue: true,
      });
      console.log(r.result.value);
    } else if (cmd === 'front') {
      await c.call('Page.bringToFront');
      console.log('已置顶当前标签页');
    } else if (cmd === 'shot') {
      const r = await c.call('Page.captureScreenshot', { format: 'png' });
      const p = args[0] || 'cdp_shot.png';
      fs.writeFileSync(p, Buffer.from(r.data, 'base64'));
      console.log('已保存: ' + p);
    } else {
      throw new Error('未知命令: ' + cmd);
    }
  });
}

main().catch((e) => { console.error('错误: ' + e.message); process.exit(1); });
