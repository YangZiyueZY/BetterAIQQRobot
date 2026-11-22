/* ============================================================
 * QQ 机器人控制台 —— 纯静态前端
 * 无框架、无构建步骤、无外部依赖
 * ============================================================ */
'use strict';

(function () {

  /* ==========================================================
   * 0. 基础工具
   * ========================================================== */

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  function esc(v) {
    return String(v === null || v === undefined ? '' : v)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function pad2(n) { return (n < 10 ? '0' : '') + n; }

  /** 时间戳 → MM-DD HH:mm:ss */
  function fmtTime(ts) {
    if (!ts) return '—';
    var d = new Date(Number(ts) * 1000);
    if (isNaN(d.getTime())) return '—';
    return pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) + ' ' +
      pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
  }

  /** 秒 → 人类可读时长 */
  function fmtDuration(sec) {
    sec = Math.max(0, Math.floor(Number(sec) || 0));
    var d = Math.floor(sec / 86400);
    var h = Math.floor((sec % 86400) / 3600);
    var m = Math.floor((sec % 3600) / 60);
    var s = sec % 60;
    if (d > 0) return d + ' 天 ' + h + ' 小时 ' + m + ' 分';
    if (h > 0) return h + ' 小时 ' + m + ' 分';
    if (m > 0) return m + ' 分 ' + s + ' 秒';
    return s + ' 秒';
  }

  function fmtNum(n) {
    var v = Number(n);
    if (!isFinite(v)) return '0';
    return v.toLocaleString('zh-CN');
  }

  function fmtPct(x, digits) {
    var v = Number(x);
    if (!isFinite(v)) return '—';
    return (v * 100).toFixed(digits === undefined ? 2 : digits) + '%';
  }

  function pctWidth(a, b) {
    var v = Number(a) || 0, t = Number(b) || 0;
    if (t <= 0) return 0;
    return Math.max(0, Math.min(100, (v / t) * 100));
  }

  var PRESENCE_LABEL = {
    online: '在线', busy: '忙碌', away: '离开', sleeping: '睡眠'
  };

  /* ---------------- toast ---------------- */

  var toastLayer = null;

  function toast(msg, type, ms) {
    if (!toastLayer) toastLayer = $('#toast-layer');
    if (!toastLayer) return;
    var el = document.createElement('div');
    el.className = 'toast ' + (type || 'info');
    el.textContent = String(msg);
    toastLayer.appendChild(el);
    var life = ms || (type === 'error' ? 6000 : 3200);
    setTimeout(function () {
      el.style.transition = 'opacity .2s ease';
      el.style.opacity = '0';
      setTimeout(function () {
        if (el.parentNode) el.parentNode.removeChild(el);
      }, 220);
    }, life);
  }

  /* ---------------- API 层 ---------------- */

  function ApiError(message, status, data) {
    this.name = 'ApiError';
    this.message = message;
    this.status = status || 0;
    this.data = data || null;
  }
  ApiError.prototype = Object.create(Error.prototype);

  function api(path, opts) {
    opts = opts || {};
    var init = {
      method: opts.method || 'GET',
      credentials: 'same-origin',
      headers: {}
    };
    if (opts.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.body);
    }

    return fetch(path, init).then(function (res) {
      if (res.status === 401) {
        handleUnauthorized();
        throw new ApiError('未认证或登录已过期', 401, null);
      }
      if (res.status === 204) {
        if (!res.ok) throw new ApiError('请求失败（HTTP ' + res.status + '）', res.status, null);
        return null;
      }
      return res.text().then(function (txt) {
        var data = null;
        if (txt) {
          try { data = JSON.parse(txt); } catch (e) { data = null; }
        }
        if (!res.ok) {
          var msg = (data && data.error) ? data.error : ('请求失败（HTTP ' + res.status + '）');
          throw new ApiError(msg, res.status, data);
        }
        return data;
      });
    }, function (e) {
      throw new ApiError('网络请求失败：' + ((e && e.message) || '未知错误'), 0, null);
    });
  }

  /** 统一错误上报：401 已由 api() 处理，其余弹 toast */
  function reportError(e, prefix) {
    if (e && e.status === 401) return;
    var msg = (e && e.message) ? e.message : '未知错误';
    toast((prefix ? prefix + '：' : '') + msg, 'error');
  }

  function setBusy(btn, busy, busyText, idleText) {
    if (!btn) return;
    if (busy) {
      if (!btn.dataset.idleText) btn.dataset.idleText = btn.textContent;
      btn.disabled = true;
      btn.textContent = busyText || '处理中…';
    } else {
      btn.disabled = false;
      btn.textContent = idleText || btn.dataset.idleText || btn.textContent;
    }
  }

  /* ==========================================================
   * 1. 认证
   * ========================================================== */

  var appVersion = '—';
  var booted = false;

  function showLogin() {
    var login = $('#login-screen');
    var app = $('#app');
    if (app) app.classList.add('hidden');
    if (login) login.classList.remove('hidden');
    var input = $('#login-token');
    if (input) { input.value = ''; input.focus(); }
  }

  function showApp() {
    var login = $('#login-screen');
    var app = $('#app');
    if (login) login.classList.add('hidden');
    if (app) app.classList.remove('hidden');
  }

  function setLoginError(msg) {
    var box = $('#login-error');
    if (!box) return;
    if (!msg) { box.classList.add('hidden'); box.textContent = ''; return; }
    box.textContent = msg;
    box.classList.remove('hidden');
  }

  function handleUnauthorized() {
    // 只有"曾经登录成功过"才算会话过期；首次访问 /api/me 返回 401 是正常的，
    // 不该吓唬用户说登录失效。
    var wasLoggedIn = booted;
    stopDashboardTimer();
    closeLogStream();
    booted = false;
    showLogin();
    setLoginError(wasLoggedIn ? '登录已失效，请重新输入访问令牌。' : '');
  }

  function bindLogin() {
    var form = $('#login-form');
    if (!form) return;
    form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var input = $('#login-token');
      var btn = $('#login-submit');
      var token = input ? input.value.trim() : '';
      if (!token) { setLoginError('请输入访问令牌。'); return; }
      setLoginError('');
      setBusy(btn, true, '登录中…');
      api('/api/login', { method: 'POST', body: { token: token } })
        .then(function () {
          setBusy(btn, false, null, '登录');
          booted = false;
          return boot();
        })
        .catch(function (e) {
          setBusy(btn, false, null, '登录');
          if (e && e.status === 401) {
            setLoginError('访问令牌不正确。');
          } else {
            setLoginError((e && e.message) || '登录失败。');
          }
        });
    });
  }

  function bindLogout() {
    var btn = $('#logout-btn');
    if (!btn) return;
    btn.addEventListener('click', function () {
      setBusy(btn, true, '退出中…');
      api('/api/logout', { method: 'POST' })
        .catch(function () { /* 退出失败也直接回到登录页 */ })
        .then(function () {
          setBusy(btn, false, null, '退出登录');
          handleUnauthorized();
        });
    });
  }

  /* ==========================================================
   * 2. 视图切换
   * ========================================================== */

  var VIEW_META = {
    dashboard: { title: '概览', sub: '机器人与消息流水线总览' },
    groups: { title: '群管理', sub: '启用开关与群内额度使用情况' },
    timeline: { title: '时间线', sub: '查看指定群的最近消息' },
    decisions: { title: '决策记录', sub: '每次发言判定的得分、原因与草稿' },
    prompt: { title: '提示词', sub: '模型实际收到的全部提示词，可查看、修改、恢复默认' },
    persona: { title: '人设', sub: '编辑 YAML 人设并预览解析结果' },
    profile: { title: '群档案', sub: '机器人对这个群的理解，以及它学到的策略' },
    memory: { title: '记忆', sub: '每个群单独记下的东西，以及它们什么时候会忘' },
    stickers: { title: '表情库', sub: '自动收藏的表情包、QQ 内置表情与收藏规则' },
    settings: { title: '设置', sub: 'LLM、行为、作息与安全参数' },
    logs: { title: '日志', sub: '实时日志流' }
  };

  var currentView = '';

  var VIEW_LOADERS = {
    dashboard: loadDashboard,
    groups: loadGroups,
    timeline: loadTimeline,
    decisions: loadDecisions,
    persona: loadPersona,
    prompt: loadPrompts,
    profile: loadProfile,
    memory: loadMemories,
    stickers: loadStickers,
    settings: loadSettings,
    logs: loadLogs
  };

  function parseHash() {
    var name = (window.location.hash || '').replace(/^#\/?/, '');
    return VIEW_LOADERS[name] ? name : 'dashboard';
  }

  function showView(name) {
    if (!VIEW_LOADERS[name]) name = 'dashboard';
    currentView = name;

    // 同步到地址栏，刷新和后退都能回到同一个视图。
    // 注意：下面 hashchange 里会先比对 currentView，所以这里不会自激循环。
    var want = '#' + name;
    if (window.location.hash !== want) window.location.hash = want;

    $$('.view').forEach(function (v) {
      v.classList.toggle('active', v.id === 'view-' + name);
    });
    $$('#nav .nav-item').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-view') === name);
    });

    var meta = VIEW_META[name];
    var t = $('#view-title'), s = $('#view-sub');
    if (t) t.textContent = meta.title;
    if (s) s.textContent = meta.sub;

    // 状态轮询常驻（连接胶囊在每个视图都要准），
    // 切到概览时立刻拉一次，不必等下一个 5 秒周期。
    if (name === 'dashboard') {
      startDashboardTimer();
      refreshDashboardStatus();
    }

    try {
      VIEW_LOADERS[name]();
    } catch (e) {
      reportError(e, '加载失败');
    }
  }

  function bindNav() {
    $$('#nav .nav-item').forEach(function (btn) {
      btn.addEventListener('click', function () {
        showView(btn.getAttribute('data-view'));
      });
    });
  }

  /* ---------------- 连接状态胶囊 ---------------- */

  function renderConnPill(connected, nickname) {
    var pill = $('#conn-pill');
    if (!pill) return;
    pill.innerHTML = '<span class="dot ' + (connected ? 'dot-ok pulse' : 'dot-bad') + '"></span>' +
      '<span>' + (connected ? '已连接' : '未连接') + '</span>';
    pill.title = nickname ? ('账号：' + nickname) : '';
  }

  /* ==========================================================
   * 3. 概览
   * ========================================================== */

  var dashboardTimer = null;

  function startDashboardTimer() {
    if (dashboardTimer !== null) return;   // 全局唯一，登录后启动一次
    dashboardTimer = window.setInterval(function () {
      refreshDashboardStatus();
    }, 5000);
  }

  function stopDashboardTimer() {
    if (dashboardTimer !== null) {
      window.clearInterval(dashboardTimer);
      dashboardTimer = null;
    }
  }

  function loadDashboard() {
    refreshDashboardStatus();
    loadTimeseries();
  }

  var statusFailNotified = false;

  /* 状态轮询是全局的：连接胶囊在任何视图都必须准。
     概览专属的那几块只在概览页渲染。 */
  function refreshDashboardStatus() {
    return api('/api/status').then(function (st) {
      if (!st) return;
      statusFailNotified = false;
      renderConnPill(!!st.connected, st.nickname);
      renderDebugBanner(st.debug);
      if (currentView !== 'dashboard') return;
      renderStatusBar(st);
      renderStatCards(st);
      renderDashMeta(st);
    }).catch(function (e) {
      renderConnPill(false);
      // 自动刷新每 5 秒一次，失败只在首次提示，避免刷屏
      if (!statusFailNotified) {
        statusFailNotified = true;
        reportError(e, '状态获取失败');
      }
    });
  }

  function renderDebugBanner(dbg) {
    var box = $('#debug-banner');
    if (!box) return;
    box.classList.toggle('hidden', !(dbg && dbg.enabled));
    if (dbg && dbg.enabled) {
      var label = box.querySelector('span');
      if (label) {
        label.innerHTML = '<b>测试模式已开启</b> —— ' +
          (dbg.affects || []).join('、') + '全部失效' +
          (dbg.respond_to_all ? '，<b>并且会接每一句话（会刷屏）</b>' : '') +
          '。测完请关掉。';
      }
    }
  }

  function renderStatusBar(st) {
    var box = $('#status-bar');
    if (!box) return;

    var presence = PRESENCE_LABEL[st.presence] || (st.presence || '未知');
    var paused = !!st.paused;
    var runState = paused ? '已暂停' : '运行中';

    box.innerHTML =
      '<span class="status-item"><span class="dot ' + (st.connected ? 'dot-ok pulse' : 'dot-bad') + '"></span>' +
      (st.connected ? '已连接' : '未连接') + '</span>' +
      '<span class="status-sep"></span>' +
      '<span class="status-item"><span class="status-name">' + esc(st.nickname || '未登录') + '</span>' +
      '<span class="faint mono small">' + esc(st.self_id || '—') + '</span></span>' +
      '<span class="status-sep"></span>' +
      '<span class="status-item">运行时长 <strong>' + esc(fmtDuration(st.uptime_seconds)) + '</strong></span>' +
      '<span class="status-sep"></span>' +
      '<span class="status-item">状态 <strong>' + esc(presence) + '</strong></span>' +
      '<span class="status-sep"></span>' +
      '<span class="status-item">调度 <strong>' + runState + '</strong></span>' +
      '<span class="status-actions">' +
      '<button type="button" class="btn ' + (paused ? 'btn-primary' : 'btn-secondary') + ' btn-sm" id="toggle-run">' +
      (paused ? '恢复' : '暂停') + '</button>' +
      '</span>';

    var btn = $('#toggle-run');
    if (btn) {
      btn.addEventListener('click', function () {
        var target = !paused; // 当前暂停 → 恢复
        var url = paused ? '/api/control/resume' : '/api/control/pause';
        setBusy(btn, true, paused ? '恢复中…' : '暂停中…');
        api(url, { method: 'POST' })
          .then(function (r) {
            setBusy(btn, false);
            toast(r && r.paused ? '已暂停发言' : '已恢复发言', 'success');
            return refreshDashboardStatus();
          })
          .catch(function (e) {
            setBusy(btn, false);
            reportError(e, target ? '暂停失败' : '恢复失败');
          });
      });
    }
  }

  function renderStatCards(st) {
    var grid = $('#stat-grid');
    if (!grid) return;
    var s = st.stats || {};

    var cards = [
      { label: '收到消息', value: fmtNum(s.messages_received), foot: '去重丢弃 ' + fmtNum(s.duplicates_dropped) },
      { label: '发出消息', value: fmtNum(s.messages_sent), foot: '回复次数 ' + fmtNum(s.replies) },
      { label: 'LLM 调用', value: fmtNum(s.llm_calls), foot: 'token ' + fmtNum(s.llm_tokens_in) + ' / ' + fmtNum(s.llm_tokens_out) },
      { label: '发言率', value: fmtPct(s.spoke_rate, 2), foot: '回复 / 决策' },
      { label: '决策次数', value: fmtNum(s.decisions), foot: '评分判定总数' },
      { label: '队列长度', value: fmtNum(s.queue_size), foot: '待处理批次' }
    ];

    grid.innerHTML = cards.map(function (c) {
      return '<div class="stat-card">' +
        '<div class="stat-label">' + esc(c.label) + '</div>' +
        '<div class="stat-value">' + esc(c.value) + '</div>' +
        '<div class="stat-foot">' + esc(c.foot) + '</div>' +
        '</div>';
    }).join('');
  }

  function renderDashMeta(st) {
    var box = $('#dash-meta');
    if (!box) return;
    var presence = PRESENCE_LABEL[st.presence] || (st.presence || '未知');
    var enabled = Number(st.groups_enabled) || 0;
    var total = Number(st.groups_total) || 0;

    var rows = [
      ['状态机状态', presence + '（' + (st.presence || '—') + '）'],
      ['已启用群数', enabled + ' / ' + total],
      ['调度开关', st.paused ? '已暂停' : '运行中'],
      ['启动时间', fmtTime(st.started_at)],
      ['运行时长', fmtDuration(st.uptime_seconds)],
      ['版本', st.version || '—']
    ];

    var html = rows.map(function (r) {
      return '<div class="kv-row"><span class="kv-key">' + esc(r[0]) + '</span>' +
        '<span class="kv-val">' + esc(r[1]) + '</span></div>';
    }).join('');

    var ratio = pctWidth(enabled, total);
    html += '<div class="kv-row"><span class="kv-key">群启用比例</span>' +
      '<span class="kv-val">' + esc(ratio.toFixed(0)) + '%</span></div>' +
      '<div class="bar-mini" style="width:100%"><i style="width:' + ratio.toFixed(1) + '%"></i></div>';

    box.innerHTML = html;
  }

  function loadTimeseries() {
    return api('/api/stats/timeseries?hours=24').then(function (data) {
      renderChart((data && data.points) || []);
    }).catch(function (e) {
      reportError(e, '24 小时统计获取失败');
      var bars = $('#chart-bars');
      if (bars) bars.innerHTML = '<div class="empty" style="width:100%">统计数据不可用</div>';
    });
  }

  function renderChart(points) {
    var bars = $('#chart-bars');
    var axis = $('#chart-axis');
    var legend = $('#chart-legend');
    if (!bars) return;

    if (!points.length) {
      bars.innerHTML = '<div class="empty" style="width:100%">暂无数据</div>';
      if (axis) axis.innerHTML = '';
      if (legend) legend.textContent = '—';
      return;
    }

    var maxTotal = 1;
    var sumIn = 0, sumOut = 0;
    points.forEach(function (p) {
      var i = Number(p.in) || 0, o = Number(p.out) || 0;
      sumIn += i; sumOut += o;
      maxTotal = Math.max(maxTotal, i + o);
    });

    bars.innerHTML = points.map(function (p) {
      var i = Number(p.in) || 0, o = Number(p.out) || 0;
      var hi = i > 0 ? Math.max(2, (i / maxTotal) * 100) : 0;
      var ho = o > 0 ? Math.max(2, (o / maxTotal) * 100) : 0;
      var label = fmtTime(p.ts).slice(6);
      return '<div class="chart-col" title="' + esc(fmtTime(p.ts) + ' 收到 ' + i + ' / 发出 ' + o) + '">' +
        '<div class="chart-bar in" style="height:' + hi.toFixed(1) + '%"></div>' +
        '<div class="chart-bar out" style="height:' + ho.toFixed(1) + '%"></div>' +
        '</div>';
    }).join('');

    if (axis) {
      // 只每 4 格标一次，并给有文字的刻度双倍宽度 —— 24 等分时每格
      // 只有 ~26px，放不下 "21:00"，会被 overflow:hidden 裁掉末位。
      axis.innerHTML = points.map(function (p, idx) {
        if (idx % 4 !== 0) return '<span></span>';
        return '<span class="tick">' + esc(fmtTime(p.ts).slice(6, 11)) + '</span>';
      }).join('');
    }
    if (legend) {
      legend.textContent = '收到 ' + fmtNum(sumIn) + ' · 发出 ' + fmtNum(sumOut) + '（深色=收到，橙色=发出）';
    }
  }

  /* ==========================================================
   * 4. 群管理
   * ========================================================== */

  var groupsCache = [];

  function loadGroups() {
    var tbody = $('#groups-body');
    if (tbody) tbody.innerHTML = '<tr><td colspan="7" class="muted">加载中…</td></tr>';
    return api('/api/groups').then(function (list) {
      groupsCache = Array.isArray(list) ? list : [];
      renderGroups(groupsCache);
      fillGroupSelect($('#tl-group'), '请选择群', groupsCache);
      fillGroupSelect($('#dec-group'), '全部群', groupsCache, true);
    }).catch(function (e) {
      reportError(e, '群列表获取失败');
      if (tbody) tbody.innerHTML = '';
      var empty = $('#groups-empty');
      if (empty) { empty.textContent = '群列表加载失败'; empty.classList.remove('hidden'); }
    });
  }

  function fillGroupSelect(sel, placeholder, list, keepAllOption) {
    if (!sel) return;
    var prev = sel.value;
    var html = '<option value="">' + esc(placeholder) + '</option>';
    html += list.map(function (g) {
      return '<option value="' + esc(g.group_id) + '">' +
        esc((g.name || '未命名群') + '（' + g.group_id + '）') + '</option>';
    }).join('');
    sel.innerHTML = html;
    if (prev) sel.value = prev;
    if (!sel.value && !keepAllOption && list.length) sel.value = String(list[0].group_id);
  }

  function renderGroups(list) {
    var tbody = $('#groups-body');
    var empty = $('#groups-empty');
    var count = $('#groups-count');
    if (!tbody) return;

    if (count) count.textContent = '共 ' + list.length + ' 个群';

    if (!list.length) {
      tbody.innerHTML = '';
      if (empty) { empty.textContent = '暂无群数据'; empty.classList.remove('hidden'); }
      return;
    }
    if (empty) empty.classList.add('hidden');

    tbody.innerHTML = list.map(function (g) {
      return '<tr data-gid="' + esc(g.group_id) + '">' +
        '<td class="t-mono nowrap">' + esc(g.group_id) + '</td>' +
        '<td>' + esc(g.name || '未命名群') + '</td>' +
        '<td><label class="switch switch-sm">' +
        '<input type="checkbox" class="g-toggle"' + (g.enabled ? ' checked' : '') + '>' +
        '<span class="switch-track"><span class="switch-thumb"></span></span>' +
        '</label></td>' +
        '<td class="t-mono">' + esc(fmtNum(g.msg_count)) + '</td>' +
        '<td class="nowrap">本时 ' + esc(fmtNum(g.hour_used)) + ' / 今日 ' + esc(fmtNum(g.day_used)) + '</td>' +
        '<td class="t-mono">' + esc(fmtNum(g.member_count)) + '</td>' +
        '<td class="nowrap muted">' + esc(fmtTime(g.last_seen)) + '</td>' +
        '</tr>';
    }).join('');

    $$('.g-toggle', tbody).forEach(function (input) {
      input.addEventListener('change', function () {
        onToggleGroup(input);
      });
    });
  }

  function onToggleGroup(input) {
    var tr = input.closest('tr');
    var gid = tr ? tr.getAttribute('data-gid') : '';
    var wanted = input.checked;
    input.disabled = true;

    api('/api/groups/' + encodeURIComponent(gid), {
      method: 'PATCH',
      body: { enabled: wanted }
    }).then(function (updated) {
      input.disabled = false;
      var obj = updated || {};
      if (typeof obj.enabled === 'boolean') input.checked = obj.enabled;
      // 同步缓存
      for (var i = 0; i < groupsCache.length; i++) {
        if (String(groupsCache[i].group_id) === String(gid)) {
          groupsCache[i].enabled = input.checked;
          break;
        }
      }
      toast('群 ' + gid + ' 已' + (input.checked ? '启用' : '停用'), 'success');
    }).catch(function (e) {
      input.disabled = false;
      input.checked = !wanted;
      reportError(e, '开关更新失败');
    });
  }

  function bindGroups() {
    var btn = $('#groups-refresh');
    if (btn) btn.addEventListener('click', function () { loadGroups(); });
  }

  /* ==========================================================
   * 5. 时间线
   * ========================================================== */

  var tlBeforeId = null;
  var tlLoaded = 0;

  function loadTimeline() {
    if (!groupsCache.length) {
      api('/api/groups').then(function (list) {
        groupsCache = Array.isArray(list) ? list : [];
        fillGroupSelect($('#tl-group'), '请选择群', groupsCache);
        fetchTimeline(false);
      }).catch(function (e) { reportError(e, '群列表获取失败'); });
      return;
    }
    fetchTimeline(false);
  }

  function fetchTimeline(append) {
    var sel = $('#tl-group');
    var gid = sel ? sel.value : '';
    var list = $('#tl-list');
    var more = $('#tl-more');
    var empty = $('#tl-empty');

    if (!gid) {
      if (list) list.innerHTML = '';
      if (more) more.classList.add('hidden');
      if (empty) { empty.textContent = '请先选择一个群'; empty.classList.remove('hidden'); }
      return Promise.resolve();
    }

    var url = '/api/groups/' + encodeURIComponent(gid) + '/messages?limit=50';
    if (append && tlBeforeId !== null) url += '&before_id=' + encodeURIComponent(tlBeforeId);

    if (more) setBusy(more, true, '加载中…');

    return api(url).then(function (data) {
      var msgs = (data && data.messages) || [];
      if (!append && list) list.innerHTML = '';
      if (empty) empty.classList.toggle('hidden', msgs.length > 0 || append);

      if (msgs.length) {
        var minId = null;
        var html = msgs.map(function (m) {
          var id = Number(m.message_id);
          if (!isNaN(id) && (minId === null || id < minId)) minId = id;
          return '<div class="msg' + (m.is_self ? ' self' : '') + '">' +
            '<div class="msg-head">' +
            '<span class="msg-name">' + esc(m.nickname || ('用户 ' + m.user_id)) +
            (m.is_self ? '（我）' : '') +
            (m.mentions_self ? '<span class="msg-mention">@我</span>' : '') +
            '</span>' +
            '<span class="msg-time">' + esc(fmtTime(m.ts)) + '</span>' +
            '</div>' +
            '<div class="msg-body">' + esc(m.text) + '</div>' +
            '</div>';
        }).join('');

        if (list) list.insertAdjacentHTML('beforeend', html);
        tlLoaded += msgs.length;
        if (minId !== null) tlBeforeId = minId;
      }

      if (more) {
        setBusy(more, false, null, '加载更多');
        more.classList.toggle('hidden', msgs.length < 50);
      }
    }).catch(function (e) {
      if (more) setBusy(more, false, null, '加载更多');
      reportError(e, '消息获取失败');
    });
  }

  function bindTimeline() {
    var sel = $('#tl-group');
    if (sel) {
      sel.addEventListener('change', function () {
        tlBeforeId = null;
        tlLoaded = 0;
        fetchTimeline(false);
      });
    }
    var refresh = $('#tl-refresh');
    if (refresh) {
      refresh.addEventListener('click', function () {
        tlBeforeId = null;
        tlLoaded = 0;
        fetchTimeline(false);
      });
    }
    var more = $('#tl-more');
    if (more) {
      more.addEventListener('click', function () { fetchTimeline(true); });
    }
  }

  /* ==========================================================
   * 6. 决策记录
   * ========================================================== */

  function loadDecisions() {
    var tbody = $('#decisions-body');
    if (tbody) tbody.innerHTML = '<tr><td colspan="7" class="muted">加载中…</td></tr>';

    var gidSel = $('#dec-group');
    var gid = gidSel ? gidSel.value : '';
    var limitSel = $('#dec-limit');
    var limit = limitSel ? limitSel.value : '50';

    var url = '/api/decisions?limit=' + encodeURIComponent(limit);
    if (gid) url += '&group_id=' + encodeURIComponent(gid);

    return api(url).then(function (list) {
      renderDecisions(Array.isArray(list) ? list : []);
      if (!groupsCache.length) {
        api('/api/groups').then(function (g) {
          groupsCache = Array.isArray(g) ? g : [];
          fillGroupSelect($('#dec-group'), '全部群', groupsCache, true);
          if (gidSel) gidSel.value = gid;
        }).catch(function () { /* 静默：群列表仅用于筛选下拉 */ });
      }
    }).catch(function (e) {
      reportError(e, '决策记录获取失败');
      if (tbody) tbody.innerHTML = '';
      var empty = $('#decisions-empty');
      if (empty) { empty.textContent = '决策记录加载失败'; empty.classList.remove('hidden'); }
    });
  }

  function renderDecisions(list) {
    var tbody = $('#decisions-body');
    var empty = $('#decisions-empty');
    if (!tbody) return;
    if (!list.length) {
      tbody.innerHTML = '';
      if (empty) { empty.textContent = '暂无决策记录'; empty.classList.remove('hidden'); }
      return;
    }
    if (empty) empty.classList.add('hidden');

    tbody.innerHTML = list.map(function (d) {
      var on = !!d.should_reply;
      var score = Number(d.priority);
      var scoreTxt = isFinite(score) ? score.toFixed(2) : '—';
      return '<tr class="' + (on ? 'row-on' : 'row-off') + '">' +
        '<td class="t-mono nowrap">' + esc(fmtTime(d.ts)) + '</td>' +
        '<td class="t-mono nowrap">' + esc(d.group_id) + '</td>' +
        '<td class="cell-flag"><span class="tag ' + (on ? 'tag-on' : '') + '">' +
        (on ? '发言' : '沉默') + '</span></td>' +
        '<td class="cell-score t-mono">' + esc(scoreTxt) + '</td>' +
        '<td>' + esc(d.reason || '—') + '</td>' +
        '<td class="cell-text">' + esc(d.text || '—') + '</td>' +
        '<td class="cell-draft">' + (d.draft ? esc(d.draft) : '<span class="faint">—</span>') + '</td>' +
        '</tr>';
    }).join('');
  }

  function bindDecisions() {
    var refresh = $('#dec-refresh');
    if (refresh) refresh.addEventListener('click', function () { loadDecisions(); });
    var sel = $('#dec-group');
    if (sel) sel.addEventListener('change', function () { loadDecisions(); });
    var lim = $('#dec-limit');
    if (lim) lim.addEventListener('change', function () { loadDecisions(); });
  }

  /* ==========================================================
   * 7. 人设
   * ========================================================== */

  /** 当前选中的人设作用域：'' = 全局默认，数字 = 某个群。 */
  var personaGroup = '';

  function loadPersona() {
    var ta = $('#persona-raw');
    if (ta) ta.value = '';
    var q = personaGroup ? ('?group_id=' + encodeURIComponent(personaGroup)) : '';
    return api('/api/persona' + q).then(function (p) {
      p = p || {};
      if (ta) ta.value = p.raw || '';
      renderPersonaSummary(p.summary);
      renderPersonaState(p.valid, p.error);
      renderPersonaTree(p.raw || '');
      renderPersonaScope(p);
    }).catch(function (e) {
      reportError(e, '人设加载失败');
      renderPersonaState(false, (e && e.message) || '人设加载失败');
    });
  }

  function renderPersonaScope(p) {
    var sel = $('#persona-group');
    var state = $('#persona-scope-state');
    var hint = $('#persona-scope-hint');
    var btn = $('#persona-reset');
    var isGroup = !!p.group_id;

    if (sel) {
      // 只在选项集合变化时重建，避免打字时选择被重置
      var want = '<option value="">🌐 全局默认（所有群）</option>';
      (p.groups || []).forEach(function (g) {
        want += '<option value="' + esc(g.group_id) + '">'
          + esc((g.custom ? '★ ' : '') + g.name + '（' + g.group_id + '）') + '</option>';
      });
      if (sel.innerHTML !== want) sel.innerHTML = want;
      sel.value = p.group_id ? String(p.group_id) : '';
    }

    if (state) {
      state.textContent = isGroup
        ? (p.has_custom ? '该群使用自定义人设' : '该群跟随全局默认')
        : '编辑的是全局默认，' + (p.custom_count || 0) + ' 个群有独立人设';
      state.style.color = (isGroup && p.has_custom) ? 'var(--accent)' : '';
    }
    if (hint) {
      hint.textContent = isGroup
        ? (p.has_custom
           ? '这份人设只对这个群生效。改完保存即可，不影响其他群。'
           : '这个群还没有自定义人设，现在看到的是全局默认。改完点保存就会创建一份专属的。')
        : '默认所有群共用这一份。想让某个群单独用一份，在上面的下拉框里选它。';
    }
    if (btn) btn.classList.toggle('hidden', !(isGroup && p.has_custom));
  }

  function renderPersonaState(valid, error) {
    var state = $('#persona-state');
    var box = $('#persona-error');
    if (state) {
      state.textContent = valid ? '校验通过' : '校验未通过';
      state.style.color = valid ? 'var(--success)' : 'var(--danger)';
    }
    if (box) {
      if (valid) {
        box.classList.add('hidden');
        box.textContent = '';
      } else {
        box.textContent = error || '未知错误';
        box.classList.remove('hidden');
      }
    }
  }

  function renderPersonaSummary(summary) {
    var chips = $('#persona-chips');
    var text = $('#persona-summary');
    if (!chips) return;
    if (!summary) {
      chips.innerHTML = '';
      if (text) text.textContent = '—';
      return;
    }
    if (text) text.textContent = summary.name ? ('名称 ' + summary.name) : '已解析';
    var items = [
      ['名称', summary.name],
      ['traits', summary.traits],
      ['few_shot', summary.few_shot],
      ['expert', summary.expert]
    ];
    chips.innerHTML = items.map(function (it) {
      var v = (it[1] === null || it[1] === undefined) ? '—' : it[1];
      return '<span class="chip">' + esc(it[0]) + ' <b>' + esc(v) + '</b></span>';
    }).join('');
  }

  /** 极简 YAML 大纲解析：只做缩进分层，用于预览，不追求完备 */
  function parseYamlOutline(text) {
    var root = { children: [], leaves: [] };
    var stack = [{ indent: -1, node: root }];
    var lines = String(text || '').split(/\r?\n/);

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line.trim()) continue;
      if (/^\s*#/.test(line)) continue;

      var indent = line.length - line.replace(/^\s+/, '').length;
      var content = line.trim();

      while (stack.length > 1 && indent <= stack[stack.length - 1].indent) stack.pop();
      var parent = stack[stack.length - 1].node;

      if (content === '-' || content.indexOf('- ') === 0) {
        if (!parent.items) parent.items = [];
        parent.items.push(content === '-' ? '' : content.slice(2).trim());
        continue;
      }

      var idx = content.indexOf(':');
      if (idx === -1) {
        parent.leaves.push({ key: '', value: content });
        continue;
      }

      var key = content.slice(0, idx).trim();
      var rest = content.slice(idx + 1).trim();

      if (rest === '' || rest === '|' || rest === '>' || rest === '|-' || rest === '>-') {
        var node = { key: key, children: [], leaves: [], items: [], block: rest !== '' };
        parent.children.push(node);
        stack.push({ indent: indent, node: node });
      } else {
        parent.leaves.push({ key: key, value: rest });
      }
    }
    return root;
  }

  function valueClass(v) {
    var s = String(v);
    if (/^-?\d+(\.\d+)?$/.test(s)) return 'num';
    if (/^(true|false|yes|no)$/i.test(s)) return 'bool';
    return '';
  }

  function renderLeaf(leaf) {
    var v = String(leaf.value == null ? '' : leaf.value);
    // 去掉包裹的引号让预览更干净
    var display = v;
    if (/^".*"$/.test(v) || /^'.*'$/.test(v)) display = v.slice(1, -1);
    return '<div class="leaf">' +
      (leaf.key ? '<span class="k">' + esc(leaf.key) + '</span>' : '') +
      '<span class="v ' + valueClass(v) + '">' + esc(display) + '</span>' +
      '</div>';
  }

  function renderNodeBody(node) {
    var html = '';
    if (node.leaves && node.leaves.length) {
      html += node.leaves.map(renderLeaf).join('');
    }
    if (node.items && node.items.length) {
      html += node.items.map(function (it) {
        return '<div class="leaf list-item"><span class="k">·</span><span class="v">' + esc(it) + '</span></div>';
      }).join('');
    }
    if (node.children && node.children.length) {
      html += node.children.map(function (c, i) {
        return renderTreeNode(c, 1, i);
      }).join('');
    }
    return html || '<div class="leaf"><span class="v faint">（空）</span></div>';
  }

  function renderTreeNode(node, depth, index) {
    var count = (node.leaves ? node.leaves.length : 0) +
      (node.items ? node.items.length : 0) +
      (node.children ? node.children.length : 0);
    var open = depth <= 1 && index < 2 ? ' open' : '';
    return '<details' + open + '>' +
      '<summary>' + esc(node.key) + '<span class="count">' + count + ' 项</span></summary>' +
      '<div class="node-body">' + renderNodeBody(node) + '</div>' +
      '</details>';
  }

  function renderPersonaTree(raw) {
    var box = $('#persona-tree');
    if (!box) return;
    if (!String(raw).trim()) {
      box.innerHTML = '<div class="empty">暂无内容</div>';
      return;
    }
    var outline;
    try {
      outline = parseYamlOutline(raw);
    } catch (e) {
      box.innerHTML = '<div class="empty">预览解析失败</div>';
      return;
    }
    var html = '';
    if (outline.leaves.length) html += outline.leaves.map(renderLeaf).join('');
    if (outline.children.length) {
      html += outline.children.map(function (c, i) { return renderTreeNode(c, 1, i); }).join('');
    }
    box.innerHTML = html || '<div class="empty">暂无可预览的结构</div>';
  }


  /* ==========================================================
   * 提示词页
   * ========================================================== */

  var promptParts = [];
  var promptGroups = [];
  var promptSel = null;       // 当前编辑的 key
  var promptPreviewTab = 'system';
  /** '' / '0' = 全局层；数字 = 群级层。 */
  var promptTier = '0';

  function loadPrompts() {
    return Promise.all([api('/api/prompts?group_id=' + encodeURIComponent(promptTier)),
                        loadPromptPreview()])
      .then(function (res) {
        var d = res[0] || {};
        promptParts = d.parts || [];
        promptGroups = d.groups || [];
        renderPromptList();
        fillPromptGroups();
        var tier = $('#prompt-tier');
        if (tier) {
          tier.textContent = (promptTier === '0')
            ? ('全局层 ' + (d.custom_global || 0) + ' 条')
            : ('群级 ' + (d.custom_group || 0) + ' 条 · 全局 ' + (d.custom_global || 0) + ' 条');
          tier.style.visibility = 'visible';
        }
        if (!promptSel && promptParts.length) selectPrompt(promptParts[0].key);
        else if (promptSel) selectPrompt(promptSel);
      })
      .catch(function (e) { reportError(e, '提示词加载失败'); });
  }

  function fillPromptGroups() {
    var sel = $('#prompt-group');
    if (!sel) return;
    var want = '<option value="0">🌐 全局默认（所有群）</option>';
    promptGroups.forEach(function (g) {
      want += '<option value="' + esc(g.group_id) + '">' + esc(g.name) + '</option>';
    });
    if (sel.innerHTML !== want) sel.innerHTML = want;
    sel.value = promptTier;
  }

  function renderPromptList() {
    var box = $('#prompt-list');
    if (!box) return;
    var lastGroup = null;
    var html = '';
    promptParts.forEach(function (p) {
      if (p.group !== lastGroup) {
        lastGroup = p.group;
        html += '<div class="prompt-group-label">' + esc(p.group) + '</div>';
      }
      var mark = p.custom_group ? '群' : (p.custom_global ? '全' : '');
      html += '<div class="prompt-item' + (p.key === promptSel ? ' active' : '')
        + '" data-prompt-key="' + esc(p.key) + '">'
        + '<span class="dot" style="opacity:' + (p.custom ? '1' : '0') + '"></span>'
        + '<span class="tier-mark">' + esc(mark) + '</span>'
        + '<span class="prompt-item-title">' + esc(p.title) + '</span>'
        + '<span class="muted small">' + p.chars + '</span></div>';
    });
    box.innerHTML = html;
  }

  function selectPrompt(key) {
    promptSel = key;
    var p = null;
    promptParts.forEach(function (x) { if (x.key === key) p = x; });
    if (!p) return;
    renderPromptList();

    var set = function (id, v) { var el = $(id); if (el) el.textContent = v; };
    set('#prompt-title', p.title);
    set('#prompt-key', p.key);
    var note = $('#prompt-note');
    if (note) note.textContent = p.note || '';
    var ph = $('#prompt-ph');
    if (ph) {
      ph.textContent = p.placeholders && p.placeholders.length
        ? ('可用占位符：' + p.placeholders.map(function (x) {
             return '{' + x.name + '} ' + x.desc; }).join('　'))
        : '这一条没有占位符';
    }
    var ta = $('#prompt-text');
    if (ta) ta.value = p.current || '';
    var badge = $('#prompt-custom');
    if (badge) {
      badge.classList.toggle('hidden', !p.custom);
      badge.textContent = p.custom_group ? '本群已改' : (p.custom_global ? '全局已改' : '已改');
    }
    // 「恢复默认」只对**当前这一层**有效
    var own = (promptTier === '0') ? p.custom_global : p.custom_group;
    var save = $('#prompt-save');
    if (save) save.disabled = false;
    var reset = $('#prompt-reset');
    if (reset) reset.disabled = !own;
  }

  function loadPromptPreview() {
    var gid = (promptGroups[0] && promptGroups[0].group_id) || 0;
    var sel = $('#prompt-group');
    if (sel && sel.value) gid = sel.value;
    return api('/api/prompts/preview?group_id=' + encodeURIComponent(gid))
      .then(function (d) {
        d = d || {};
        var meta = $('#prompt-preview-meta');
        if (meta) {
          meta.textContent = d.error ? d.error
            : ('群 ' + d.group_id + ' · 人设「' + d.persona_name + '」'
               + (d.persona_scope === 'group' ? '（专属）' : '（默认）')
               + ' · system ' + d.system_chars + ' 字');
        }
        window.__promptPreview = d;
        renderPromptPreview();
      });
  }

  function renderPromptPreview() {
    var pre = $('#prompt-preview');
    var d = window.__promptPreview || {};
    if (!pre) return;
    pre.textContent = (promptPreviewTab === 'system' ? d.system : d.user) || '（暂无）';
  }

  function bindPrompts() {
    document.addEventListener('click', function (ev) {
      var item = ev.target.closest ? ev.target.closest('[data-prompt-key]') : null;
      if (item) { selectPrompt(item.getAttribute('data-prompt-key')); return; }
      var tab = ev.target.closest ? ev.target.closest('[data-preview]') : null;
      if (tab) {
        promptPreviewTab = tab.getAttribute('data-preview');
        $$('.preview-tabs .tab').forEach(function (t) { t.classList.remove('active'); });
        tab.classList.add('active');
        renderPromptPreview();
      }
    });

    var gsel = $('#prompt-group');
    if (gsel) {
      gsel.addEventListener('change', function () {
        promptTier = gsel.value || '0';   // '0' = 全局层
        loadPrompts();
      });
    }

    var save = $('#prompt-save');
    if (save) {
      save.addEventListener('click', function () {
        var ta = $('#prompt-text');
        if (!promptSel || !ta) return;
        setBusy(save, true, '保存中…');
        api('/api/prompts', { method: 'PUT',
          body: { key: promptSel, text: ta.value, group_id: Number(promptTier) } })
          .then(function (d) {
            setBusy(save, false, null, '保存');
            d = d || {};
            promptParts = d.parts || promptParts;
            renderPromptList(); selectPrompt(promptSel);
            toast((promptTier === '0' ? '全局层' : '本群') + '提示词已保存', 'success');
            loadPromptPreview();
          })
          .catch(function (e) {
            setBusy(save, false, null, '保存');
            var msg = $('#prompt-msg');
            if (msg) { msg.textContent = (e && e.message) || '保存失败'; msg.classList.remove('hidden'); }
            reportError(e, '保存失败');
          });
      });
    }

    var reset = $('#prompt-reset');
    if (reset) {
      reset.addEventListener('click', function () {
        if (!promptSel) return;
        api('/api/prompts/reset', { method: 'POST',
          body: { key: promptSel, group_id: Number(promptTier) } })
          .then(function (d) {
            d = d || {};
            promptParts = d.parts || promptParts;
            renderPromptList(); selectPrompt(promptSel);
            toast('已恢复默认', 'success');
            loadPromptPreview();
          })
          .catch(function (e) { reportError(e, '恢复失败'); });
      });
    }

    var resetAll = $('#prompt-reset-all');
    if (resetAll) {
      resetAll.addEventListener('click', function () {
        var where = (promptTier === '0') ? '全局层' : '这个群的群级层';
        if (!confirm('把' + where + '的全部提示词恢复成代码默认？该层的修改都会丢失。')) return;
        setBusy(resetAll, true, '恢复中…');
        api('/api/prompts/reset', { method: 'POST',
          body: { all: true, group_id: Number(promptTier) } })
          .then(function (d) {
            setBusy(resetAll, false, null, '全部恢复默认');
            d = d || {};
            promptParts = d.parts || promptParts;
            renderPromptList(); if (promptSel) selectPrompt(promptSel);
            toast('已全部恢复默认', 'success');
            loadPromptPreview();
          })
          .catch(function (e) {
            setBusy(resetAll, false, null, '全部恢复默认');
            reportError(e, '恢复失败');
          });
      });
    }
  }

  function bindPersona() {
    var sel = $('#persona-group');
    if (sel) {
      sel.addEventListener('change', function () {
        personaGroup = sel.value;
        loadPersona();
      });
    }
    var reset = $('#persona-reset');
    if (reset) {
      reset.addEventListener('click', function () {
        if (!personaGroup) return;
        if (!confirm('让这个群恢复跟随全局默认？该群的自定义人设会被删除。')) return;
        setBusy(reset, true, '恢复中…');
        api('/api/persona/reset', { method: 'POST', body: { group_id: Number(personaGroup) } })
          .then(function (p) {
            setBusy(reset, false, null, '恢复跟随默认');
            toast('已恢复跟随全局默认', 'success');
            renderPersonaScope(p || {});
            loadPersona();
          })
          .catch(function (e) {
            setBusy(reset, false, null, '恢复跟随默认');
            reportError(e, '恢复失败');
          });
      });
    }

    var validate = $('#persona-validate');
    var save = $('#persona-save');
    var ta = $('#persona-raw');

    if (validate) {
      validate.addEventListener('click', function () {
        var raw = ta ? ta.value : '';
        setBusy(validate, true, '校验中…');
        renderPersonaTree(raw);
        api('/api/persona/validate', { method: 'POST', body: { raw: raw } })
          .then(function (r) {
            setBusy(validate, false, null, '校验');
            r = r || {};
            renderPersonaState(!!r.valid, r.error);
            if (r.summary) renderPersonaSummary(r.summary);
            toast(r.valid ? '人设校验通过' : '人设校验未通过', r.valid ? 'success' : 'warn');
          })
          .catch(function (e) {
            setBusy(validate, false, null, '校验');
            renderPersonaState(false, (e && e.message) || '校验请求失败');
            reportError(e, '校验失败');
          });
      });
    }

    if (save) {
      save.addEventListener('click', function () {
        var raw = ta ? ta.value : '';
        setBusy(save, true, '保存中…');
        var body = { raw: raw };
        if (personaGroup) body.group_id = Number(personaGroup);
        api('/api/persona', { method: 'PUT', body: body })
          .then(function (r) {
            setBusy(save, false, null, '保存');
            r = r || {};
            renderPersonaState(!!r.valid, r.error);
            if (r.summary) renderPersonaSummary(r.summary);
            renderPersonaTree(raw);
            renderPersonaScope(r);
            var where = personaGroup ? '该群的专属人设' : '全局默认人设';
            toast(r.valid ? (where + '已保存') : (where + '已保存，但校验未通过'),
                  r.valid ? 'success' : 'warn');
          })
          .catch(function (e) {
            setBusy(save, false, null, '保存');
            renderPersonaState(false, (e && e.message) || '保存请求失败');
            reportError(e, '保存失败');
          });
      });
    }
  }

  /* ==========================================================
   * 8. 设置
   * ========================================================== */

  function setVal(id, v) {
    var el = $('#' + id);
    if (!el) return;
    if (v === null || v === undefined) return;
    if (el.type === 'checkbox') { el.checked = !!v; return; }
    el.value = String(v);
  }

  function numOf(id) {
    var el = $('#' + id);
    if (!el) return 0;
    var v = parseFloat(el.value);
    return isFinite(v) ? v : 0;
  }

  function intOf(id) {
    var el = $('#' + id);
    if (!el) return 0;
    var v = parseInt(el.value, 10);
    return isFinite(v) ? v : 0;
  }

  function strOf(id) {
    var el = $('#' + id);
    return el ? String(el.value) : '';
  }

  /** 把每个滑块的值写进对应的标签。
   *
   *  按约定自动配对：input#set-X <-> span#val-X。
   *  原来这里是硬编码三个滑块，新加一个就漏一个（"触发阈值"就漏了，显示成 —）。
   */
  function syncRangeLabels() {
    $$('input.range').forEach(function (el) {
      if (!el.id || el.id.indexOf('set-') !== 0) return;
      var label = $('#val-' + el.id.slice(4));
      if (label) label.textContent = Number(el.value).toFixed(2);
    });
  }

  function loadSettings() {
    var state = $('#settings-state');
    if (state) state.textContent = '加载中…';
    return api('/api/settings').then(function (s) {
      applySettings(s || {});
      if (state) state.textContent = '';
    }).catch(function (e) {
      if (state) state.textContent = '';
      reportError(e, '设置加载失败');
    });
  }

  function applySettings(s) {
    var llm = s.llm || {};
    var beh = s.behavior || {};
    var sch = s.schedule || {};
    var saf = s.safety || {};

    setVal('set-base-url', llm.base_url);
    setVal('set-model', llm.model);
    setVal('set-temperature', llm.temperature);
    setVal('set-max-tokens', llm.max_tokens);
    setVal('set-timeout', llm.timeout);

    var keyInput = $('#set-api-key');
    if (keyInput) {
      keyInput.value = '';
      keyInput.placeholder = llm.api_key_set ? '已配置（留空则不修改）' : '未配置';
    }
    var bh = s.behavior || {};
    setVal('set-typing-min', bh.typing_ms_per_char_min);
    setVal('set-typing-max', bh.typing_ms_per_char_max);
    setVal('set-typing-floor', bh.typing_min_delay);
    setVal('set-typing-ceil', bh.typing_max_delay);
    setVal('set-reaction-min', bh.reaction_delay_min);
    setVal('set-reaction-max', bh.reaction_delay_max);
    var tf = $('#set-typing-first');
    if (tf) tf.checked = bh.typing_first_bubble !== false;
    renderTypingPreview();

    var bl = s.bilibili || {};
    var blBox = $('#set-bili-enabled'); if (blBox) blBox.checked = !!bl.enabled;
    var blLink = $('#set-bili-link'); if (blLink) blLink.checked = bl.enabled_link !== false;
    var blSearch = $('#set-bili-search'); if (blSearch) blSearch.checked = bl.enabled_search !== false;
    var blAsr = $('#set-bili-asr'); if (blAsr) blAsr.checked = bl.asr_fallback !== false;
    setVal('set-bili-maxasr', bl.max_asr_seconds);
    setVal('set-bili-maxchars', bl.max_chars);
    setVal('set-bili-results', bl.search_results);
    var blCk = $('#set-bili-cookie');
    if (blCk) {
      blCk.value = '';
      blCk.placeholder = bl.cookie_set ? '已配置（留空则不修改）' : '未配置（AI 字幕会拿不到）';
    }
    var blSt = $('#bili-state');
    if (blSt) blSt.textContent = bl.enabled ? (bl.cookie_set ? '已就绪' : '缺 Cookie') : '未启用';

    var ar = s.asr || {};
    var arBox = $('#set-asr-enabled'); if (arBox) arBox.checked = !!ar.enabled;
    setVal('set-asr-model', ar.model);
    setVal('set-asr-timeout', ar.timeout);
    setVal('set-asr-hotwords', (ar.hotwords || []).join(','));
    var arKey = $('#set-asr-key');
    if (arKey) {
      arKey.value = '';
      arKey.placeholder = ar.api_key_set ? '已配置（留空则不修改）' : '未配置';
    }
    var arSt = $('#asr-state');
    if (arSt) arSt.textContent = ar.enabled ? (ar.api_key_set ? '已就绪' : '缺 API Key') : '未启用';

    var se = s.search || {};
    var seBox = $('#set-search-enabled');
    if (seBox) seBox.checked = !!se.enabled;
    setVal('set-reply-model', (s.llm || {}).reply_model);
    setVal('set-analysis-model', (s.llm || {}).analysis_model);
    setVal('set-search-provider', se.provider || 'tavily');
    setVal('set-search-base', se.base_url);
    setVal('set-search-results', se.max_results);
    setVal('set-search-timeout', se.timeout);
    setVal('set-search-threshold', se.need_threshold);
    setVal('set-search-snippet', se.max_snippet_chars);
    setVal('set-search-hourly', se.hourly_budget);
    setVal('set-search-daily', se.daily_budget);
    var seM = $('#set-search-mention'); if (seM) seM.checked = se.search_when_mention !== false;
    var seKey = $('#set-search-key');
    if (seKey) {
      seKey.value = '';
      seKey.placeholder = se.api_key_set ? '已配置（留空则不修改）' : '未配置';
    }
    var seState = $('#search-state');
    if (seState) {
      seState.textContent = se.api_key_set ? (se.enabled ? '已就绪' : '未启用') : '缺 API Key';
    }
    if (typeof syncRangeLabels === 'function') syncRangeLabels();

    var ocr = s.paddleocr || {};
    var mb = $('#set-paddle-enabled');
    if (mb) mb.checked = !!ocr.enabled;
    setVal('set-paddle-model', ocr.model);
    setVal('set-paddle-poll', ocr.poll_interval);
    setVal('set-paddle-timeout', ocr.job_timeout);
    setVal('set-paddle-maxchars', ocr.max_chars);
    setVal('set-paddle-minside', ocr.min_side);
    var oa = $('#set-paddle-orient'); if (oa) oa.checked = !!ocr.doc_orientation;
    var ou = $('#set-paddle-unwarp'); if (ou) ou.checked = !!ocr.doc_unwarping;
    var oc = $('#set-paddle-chart'); if (oc) oc.checked = !!ocr.chart_recognition;
    var ot = $('#set-paddle-token');
    if (ot) {
      ot.value = '';
      ot.placeholder = ocr.token_set ? '已配置（留空则不修改）' : '未配置';
    }
    var ocrb = $('#set-ocr-backend');
    if (ocrb) ocrb.value = (s.media && s.media.ocr_backend) || 'napcat';
    var st = $('#ocr-state');
    if (st) {
      st.textContent = ocr.configured ? '已就绪' : (ocr.enabled ? '缺 Token' : '未启用');
    }

    var dbgBox = $('#set-debug-enabled');
    if (dbgBox) dbgBox.checked = !!(s.debug && s.debug.enabled);
    var dbgAll = $('#set-debug-all');
    if (dbgAll) dbgAll.checked = !!(s.debug && s.debug.respond_to_all);
    var thinkBox = $('#set-keep-thinking');
    if (thinkBox) thinkBox.checked = llm.keep_thinking !== false;
    setVal('set-reasoning-effort', llm.reasoning_effort || 'high');
    setVal('set-reply-max-tokens', llm.reply_max_tokens);
    setVal('set-analysis-max-tokens', llm.analysis_max_tokens);
    setVal('set-reply-timeout', llm.reply_timeout);

    setVal('set-threshold', beh.threshold);
    setVal('set-give-up', beh.give_up_prob);
    setVal('set-mention-only', beh.mention_only);
    setVal('set-group-cooldown', beh.group_cooldown);
    setVal('set-debounce', beh.debounce);
    setVal('set-max-wait', beh.max_wait);
    setVal('set-max-batch', beh.max_batch);

    setVal('set-active-from', sch.active_from);
    setVal('set-active-to', sch.active_to);
    setVal('set-jitter', sch.jitter_minutes);
    setVal('set-hour-budget', sch.group_hour_budget);
    setVal('set-day-budget', sch.group_day_budget);
    setVal('set-global-budget', sch.global_day_budget);

    setVal('set-max-chars', saf.max_chars_per_bubble);
    setVal('set-max-bubbles', saf.max_bubbles);

    syncRangeLabels();
  }

  function collectSettings() {
    var payload = {
      media: {
        ocr_backend: strOf('set-ocr-backend') || 'napcat'
      },
      bilibili: (function () {
        var patch = {
          enabled: !!($('#set-bili-enabled') && $('#set-bili-enabled').checked),
          enabled_link: !!($('#set-bili-link') && $('#set-bili-link').checked),
          enabled_search: !!($('#set-bili-search') && $('#set-bili-search').checked),
          asr_fallback: !!($('#set-bili-asr') && $('#set-bili-asr').checked),
          max_asr_seconds: intOf('set-bili-maxasr'),
          max_chars: intOf('set-bili-maxchars'),
          search_results: intOf('set-bili-results')
        };
        var ck = ($('#set-bili-cookie') && $('#set-bili-cookie').value.trim()) || '';
        if (ck) patch.cookie = ck;
        return patch;
      })(),
      asr: (function () {
        var patch = {
          enabled: !!($('#set-asr-enabled') && $('#set-asr-enabled').checked),
          model: strOf('set-asr-model').trim(),
          timeout: numOf('set-asr-timeout')
        };
        var hw = ($('#set-asr-hotwords') && $('#set-asr-hotwords').value.trim()) || '';
        patch.hotwords = hw ? hw.split(/[,，\s]+/).filter(Boolean) : [];
        var k = ($('#set-asr-key') && $('#set-asr-key').value.trim()) || '';
        if (k) patch.api_key = k;
        return patch;
      })(),
      search: (function () {
        var patch = {
          enabled: !!($('#set-search-enabled') && $('#set-search-enabled').checked),
          provider: strOf('set-search-provider') || 'tavily',
          base_url: strOf('set-search-base').trim(),
          max_results: intOf('set-search-results'),
          timeout: numOf('set-search-timeout'),
          need_threshold: numOf('set-search-threshold'),
          max_snippet_chars: intOf('set-search-snippet'),
          hourly_budget: intOf('set-search-hourly'),
          daily_budget: intOf('set-search-daily'),
          search_when_mention: !!($('#set-search-mention') && $('#set-search-mention').checked)
        };
        var k = ($('#set-search-key') && $('#set-search-key').value.trim()) || '';
        if (k) patch.api_key = k;
        return patch;
      })(),
      paddleocr: (function () {
        var patch = {
          enabled: !!($('#set-paddle-enabled') && $('#set-paddle-enabled').checked),
          model: strOf('set-paddle-model').trim(),
          poll_interval: numOf('set-paddle-poll'),
          job_timeout: numOf('set-paddle-timeout'),
          max_chars: intOf('set-paddle-maxchars'),
          min_side: intOf('set-paddle-minside'),
          doc_orientation: !!($('#set-paddle-orient') && $('#set-paddle-orient').checked),
          doc_unwarping: !!($('#set-paddle-unwarp') && $('#set-paddle-unwarp').checked),
          chart_recognition: !!($('#set-paddle-chart') && $('#set-paddle-chart').checked)
        };
        var tok = ($('#set-paddle-token') && $('#set-paddle-token').value.trim()) || '';
        if (tok) patch.token = tok;      // 留空表示不修改
        return patch;
      })(),
      debug: {
        enabled: !!($('#set-debug-enabled') && $('#set-debug-enabled').checked),
        respond_to_all: !!($('#set-debug-all') && $('#set-debug-all').checked)
      },
      llm: {
        base_url: strOf('set-base-url').trim(),
        model: strOf('set-model').trim(),
        reply_model: strOf('set-reply-model').trim(),
        analysis_model: strOf('set-analysis-model').trim(),
        temperature: numOf('set-temperature'),
        max_tokens: intOf('set-max-tokens'),
        timeout: numOf('set-timeout'),
        keep_thinking: !!($('#set-keep-thinking') && $('#set-keep-thinking').checked),
        reasoning_effort: strOf('set-reasoning-effort') || 'high',
        reply_max_tokens: intOf('set-reply-max-tokens'),
        analysis_max_tokens: intOf('set-analysis-max-tokens'),
        reply_timeout: numOf('set-reply-timeout')
      },
      behavior: {
        threshold: numOf('set-threshold'),
        typing_ms_per_char_min: numOf('set-typing-min'),
        typing_ms_per_char_max: numOf('set-typing-max'),
        typing_min_delay: numOf('set-typing-floor'),
        typing_max_delay: numOf('set-typing-ceil'),
        reaction_delay_min: numOf('set-reaction-min'),
        reaction_delay_max: numOf('set-reaction-max'),
        typing_first_bubble: !!($('#set-typing-first') && $('#set-typing-first').checked),
        give_up_prob: numOf('set-give-up'),
        mention_only: !!($('#set-mention-only') && $('#set-mention-only').checked),
        group_cooldown: numOf('set-group-cooldown'),
        debounce: numOf('set-debounce'),
        max_wait: numOf('set-max-wait'),
        max_batch: intOf('set-max-batch')
      },
      schedule: {
        active_from: strOf('set-active-from'),
        active_to: strOf('set-active-to'),
        jitter_minutes: intOf('set-jitter'),
        group_hour_budget: intOf('set-hour-budget'),
        group_day_budget: intOf('set-day-budget'),
        global_day_budget: intOf('set-global-budget')
      },
      safety: {
        max_chars_per_bubble: intOf('set-max-chars'),
        max_bubbles: intOf('set-max-bubbles')
      }
    };
    var key = strOf('set-api-key').trim();
    if (key) payload.llm.api_key = key;
    return payload;
  }

  function bindSettings() {
    var form = $('#settings-form');
    if (form) {
      form.addEventListener('submit', function (ev) {
        ev.preventDefault();
        var btn = $('#settings-save');
        var state = $('#settings-state');
        setBusy(btn, true, '保存中…');
        if (state) state.textContent = '';
        api('/api/settings', { method: 'PUT', body: collectSettings() })
          .then(function (s) {
            setBusy(btn, false, null, '保存设置');
            applySettings(s || {});
            if (state) state.textContent = '已保存 ' + fmtTime(Math.floor(Date.now() / 1000));
            toast('设置已保存', 'success');
          })
          .catch(function (e) {
            setBusy(btn, false, null, '保存设置');
            reportError(e, '保存失败');
          });
      });
    }

    ['#set-temperature', '#set-threshold', '#set-give-up'].forEach(function (sel) {
      var el = $(sel);
      if (el) el.addEventListener('input', syncRangeLabels);
    });
    // 新增的滑块也要挂上 —— 别只挂几个写死的 id
    $$('input.range').forEach(function (el) {
      if (el.id) el.addEventListener('input', syncRangeLabels);
    });

    var test = $('#llm-test');
    var out = $('#llm-test-result');
    if (test) {
      test.addEventListener('click', function () {
        setBusy(test, true, '测试中…');
        if (out) { out.textContent = ''; out.style.color = ''; }
        api('/api/settings/llm/test', { method: 'POST' })
          .then(function (r) {
            setBusy(test, false, null, '测试 LLM 连接');
            r = r || {};
            if (r.ok) {
              var line = '延迟 ' + r.latency_ms + ' ms · 回复：' + (r.reply || '（空）');
              if (typeof r.reasoning_tokens === 'number') {
                line += ' · 思考 token ' + r.reasoning_tokens;
              }
              if (r.thinking) {
                line += ' · 思维链 ' + (r.reasoning_effort || '开');
              } else {
                line += ' · 思维链已关';
              }
              if (out) { out.textContent = line; out.style.color = 'var(--success)'; }
              toast('LLM 连接正常', 'success');
            } else {
              var msg = r.error || '测试失败';
              if (out) { out.textContent = msg; out.style.color = 'var(--danger)'; }
              toast('LLM 测试失败：' + msg, 'error');
            }
          })
          .catch(function (e) {
            setBusy(test, false, null, '测试 LLM 连接');
            var m = (e && e.message) || '测试请求失败';
            if (out) { out.textContent = m; out.style.color = 'var(--danger)'; }
            reportError(e, 'LLM 测试失败');
          });
      });
    }
  }

  /* ==========================================================
   * 9. 日志
   * ========================================================== */

  var logSource = null;
  var logView = null;
  var logStreamOn = false;
  var logStreamFailed = false;

  function logLineEl(item) {
    var level = String((item && item.level) || 'INFO').toUpperCase();
    var cls = 'lv-info';
    if (level === 'ERROR' || level === 'CRITICAL') cls = 'lv-error';
    else if (level === 'WARNING' || level === 'WARN') cls = 'lv-warning';
    else if (level === 'DEBUG' || level === 'TRACE') cls = 'lv-debug';

    var el = document.createElement('div');
    el.className = 'log-line ' + cls;
    el.innerHTML = '<span class="l-ts">' + esc(fmtTime(item && item.ts)) + '</span>' +
      '<span class="l-lv">' + esc(level) + '</span>' +
      '<span class="l-src">' + esc((item && item.logger) || '-') + '</span>' +
      '<span class="l-msg">' + esc((item && item.message) || '') + '</span>';
    return el;
  }

  function appendLogs(items, replace) {
    if (!logView) logView = $('#log-view');
    if (!logView) return;
    if (replace) logView.innerHTML = '';
    var frag = document.createDocumentFragment();
    (items || []).forEach(function (it) { frag.appendChild(logLineEl(it)); });
    logView.appendChild(frag);
    // 限制 DOM 规模，避免长时间运行后卡顿
    while (logView.childNodes.length > 3000) logView.removeChild(logView.firstChild);
    if ($('#log-autoscroll') && $('#log-autoscroll').checked) {
      logView.scrollTop = logView.scrollHeight;
    }
  }

  function setLogStatus(on, text) {
    var pill = $('#log-status');
    if (!pill) return;
    pill.innerHTML = '<span class="dot ' + (on ? 'dot-ok pulse' : 'dot-bad') + '"></span>' +
      '<span>' + esc(text || (on ? '已连接' : '未连接')) + '</span>';
  }

  function openLogStream() {
    if (logSource || !window.EventSource) return;
    var es = new EventSource('/api/logs/stream');
    logSource = es;

    function onLog(ev) {
      var item = null;
      try { item = JSON.parse(ev.data); } catch (e) { item = null; }
      if (!item) return;
      appendLogs([item], false);
    }

    es.addEventListener('log', onLog);
    es.addEventListener('message', onLog);

    es.addEventListener('open', function () {
      logStreamOn = true;
      logStreamFailed = false;
      setLogStatus(true, '已连接');
    });

    es.addEventListener('error', function () {
      logStreamOn = false;
      setLogStatus(false, '连接中断，重试中…');
      if (!logStreamFailed) {
        logStreamFailed = true;
        toast('日志流连接中断，浏览器会自动重连', 'warn');
      }
    });
  }

  function closeLogStream() {
    if (logSource) {
      try { logSource.close(); } catch (e) { /* ignore */ }
      logSource = null;
    }
    logStreamOn = false;
    setLogStatus(false, '未连接');
  }

  function loadLogs() {
    if (!logView) logView = $('#log-view');
    if (!logSource) {
      setLogStatus(false, '连接中…');
      api('/api/logs/recent?limit=200').then(function (data) {
        appendLogs((data && data.logs) || [], true);
        openLogStream();
      }).catch(function (e) {
        setLogStatus(false, '历史日志加载失败');
        reportError(e, '历史日志加载失败');
        openLogStream();
      });
    } else if (!logStreamOn) {
      setLogStatus(false, '连接中…');
    }
    if (logView && $('#log-autoscroll') && $('#log-autoscroll').checked) {
      logView.scrollTop = logView.scrollHeight;
    }
  }

  function bindLogs() {
    var clear = $('#log-clear');
    if (clear) {
      clear.addEventListener('click', function () {
        if (logView) logView.innerHTML = '';
      });
    }
    var auto = $('#log-autoscroll');
    if (auto) {
      auto.addEventListener('change', function () {
        if (auto.checked && logView) logView.scrollTop = logView.scrollHeight;
      });
    }
    var pill = $('#log-status');
    if (pill) {
      pill.style.cursor = 'pointer';
      pill.title = '点击重新连接日志流';
      pill.addEventListener('click', function () {
        closeLogStream();
        setLogStatus(false, '连接中…');
        openLogStream();
      });
    }
  }

  /* ==========================================================
   * 10. 启动
   * ========================================================== */

  var uiBound = false;


  /* ==========================================================
   * 9. 群档案
   * ========================================================== */

  var profileGroup = null;

  /** 单序列柱状图（复用概览页的 chart 样式） */
  function renderSingleBars(barsEl, axisEl, items, unit) {
    if (!barsEl) return;
    if (!items.length) {
      barsEl.innerHTML = '<div class="empty" style="width:100%">暂无数据</div>';
      if (axisEl) axisEl.innerHTML = '';
      return;
    }
    var max = 1;
    items.forEach(function (it) { max = Math.max(max, it.value); });

    barsEl.innerHTML = items.map(function (it) {
      var h = it.value > 0 ? Math.max(3, (it.value / max) * 100) : 0;
      return '<div class="chart-col" title="' + esc(it.title || (it.label + ' ' + it.value + (unit || ''))) + '">' +
        '<div class="chart-bar in" style="height:' + h.toFixed(1) + '%"></div></div>';
    }).join('');

    if (axisEl) {
      axisEl.innerHTML = items.map(function (it, i) {
        if (i % 4 !== 0) return '<span></span>';
        return '<span class="tick">' + esc(it.label) + '</span>';
      }).join('');
    }
  }

  function loadProfile() {
    var empty = $('#pf-empty');
    return api('/api/groups').then(function (list) {
      groupsCache = Array.isArray(list) ? list : [];
      var sel = $('#pf-group');
      if (sel) fillGroupSelect(sel, '请选择群', groupsCache);
      profileGroup = sel && sel.value ? sel.value : null;
      if (!profileGroup) {
        if (empty) { empty.textContent = '还没有已发现的群'; empty.classList.remove('hidden'); }
        return null;
      }
      if (empty) empty.classList.add('hidden');
      return loadProfileFor(profileGroup);
    }).catch(function (e) { reportError(e, '群档案加载失败'); });
  }

  function loadProfileFor(gid) {
    var q = encodeURIComponent(gid);
    return Promise.all([
      api('/api/groups/' + q + '/profile'),
      api('/api/groups/' + q + '/policy'),
      api('/api/feedback/timeseries?days=14&group_id=' + q),
      api('/api/onboard?group_id=' + q)
    ]).then(function (res) {
      renderProfile(res[0]);
      renderPolicy(res[1], gid);
      renderReward((res[2] && res[2].points) || []);
      renderOnboard(res[3] || {});
    }).catch(function (e) { reportError(e, '群档案加载失败'); });
  }


  /** 冷启动状态 + 伪装存活账本。 */
  function renderOnboard(d) {
    var ob = d.onboard || {};
    var sv = d.survival || {};
    var statusEl = $('#ob-status');
    var labels = {
      '': '未开始', idle: '未开始', pulling: '正在拉历史…',
      analyzing: '正在分析…', ready: '分析完成', failed: '分析失败'
    };
    if (statusEl) {
      statusEl.textContent = labels[ob.status] || ob.status || '—';
      statusEl.style.color = ob.status === 'failed' ? 'var(--danger)' : '';
    }

    var grid = $('#ob-stats');
    if (!grid) return;
    var verdicts = (sv.history || []).slice(0, 7).map(function (h) {
      var mark = { survived: '✓', suspected: '✗', too_quiet: '－', idle: '·' }[h.verdict] || '?';
      var color = h.verdict === 'survived' ? 'var(--success)'
        : (h.verdict === 'idle' ? 'var(--text-faint)' : 'var(--danger)');
      return '<span class="tag" title="' + esc(h.label + '｜发言 ' + h.spoke
        + ' 条｜群消息 ' + h.group_msgs + ' 条｜奖励 ' + h.reward) + '">'
        + '<b style="color:' + color + '">' + mark + '</b> '
        + esc(h.day_key.slice(5)) + '</span>';
    }).join(' ') || '<span class="muted small">还没有结算记录（按天结算，每天一次）</span>';

    grid.innerHTML = [
      { label: '历史样本', value: fmtNum(ob.pulled || 0), foot: '冷启动读过的群消息条数' },
      { label: '存活连胜', value: String(sv.streak || 0) + ' 天', foot: '连续没被识破的天数' },
      { label: '累计奖励', value: (sv.reward_total >= 0 ? '+' : '') + (sv.reward_total || 0), foot: '正=装得像，负=露馅或该说没说' },
      { label: '被质疑次数', value: fmtNum(sv.suspicion_total || 0), foot: '被怀疑是机器人的次数' },
      { label: '已结算', value: fmtNum(sv.days || 0) + ' 天', foot: '其中潜水失职会扣分' },
      { label: '待问问题', value: fmtNum(ob.pending || 0), foot: '看不懂的，会挑时机在群里问' }
    ].map(function (c) {
      return '<div class="stat-card"><div class="stat-label">' + esc(c.label) + '</div>' +
        '<div class="stat-value">' + esc(c.value) + '</div>' +
        '<div class="stat-foot">' + esc(c.foot) + '</div></div>';
    }).join('') + '<div class="stat-card" style="grid-column:1/-1"><div class="stat-label">最近结算</div>' +
      '<div style="margin-top:6px">' + verdicts + '</div></div>';

    var box = $('#ob-questions');
    if (!box) return;
    var qs = (ob.questions || []).map(function (q) {
      var asked = (ob.asked || []).some(function (a) { return a.text === q.text; });
      return '<li style="' + (asked ? 'opacity:.45' : '') + '">' + esc(q.text)
        + (q.why ? ' <span class="muted small">（' + esc(q.why) + '）</span>' : '')
        + (asked ? ' <span class="muted small">已问过</span>' : '') + '</li>';
    }).join('');
    var susp = (sv.recent_suspicions || []).map(function (r) {
      return '<li>' + esc(r.nickname || '?') + '：' + esc(r.text || '')
        + ' <span class="muted small">命中「' + esc(r.pattern || '') + '」强度 '
        + esc(String(r.strength || 0)) + '</span></li>';
    }).join('');

    box.innerHTML =
      (qs ? '<div class="prompt-title" style="margin-top:14px">它想问的问题</div><ul class="muted small" style="margin:6px 0 0 18px">' + qs + '</ul>' : '')
      + (ob.note ? '<div class="muted small" style="margin-top:10px">分析结论：' + esc(ob.note) + '</div>' : '')
      + (susp ? '<div class="prompt-title" style="margin-top:14px">最近被质疑的记录</div><ul class="muted small" style="margin:6px 0 0 18px">' + susp + '</ul>' : '');
  }

  function renderProfile(p) {
    var grid = $('#pf-stats');
    if (!grid) return;
    var topics = (p.top_topics || []).map(function (t) {
      return '<span class="tag">' + esc(t[0]) + ' ' + Math.round((t[1] || 0) * 100) + '%</span>';
    }).join(' ') || '<span class="muted small">还没识别出主题（样本不足或未复盘）</span>';

    grid.innerHTML = [
      { label: '样本消息', value: fmtNum(p.sample_size), foot: '用于统计的条数' },
      { label: '平均句长', value: (p.avg_msg_len || 0).toFixed(1), foot: '群友说多长，她就说多长' },
      { label: 'QQ 表情率', value: fmtPct(p.emoji_rate || 0), foot: '带小黄脸的消息占比' },
      { label: '表情包率', value: fmtPct(p.sticker_rate || 0), foot: '带自定义表情的占比' },
      { label: '提问率', value: fmtPct(p.question_rate || 0), foot: '决定层据此判断是否插话' },
      { label: '群氛围', value: (p.atmosphere || 0).toFixed(2), foot: '-1 冷淡 / +1 热闹' }
    ].map(function (c) {
      return '<div class="stat-card"><div class="stat-label">' + esc(c.label) + '</div>' +
        '<div class="stat-value">' + esc(c.value) + '</div>' +
        '<div class="stat-foot">' + esc(c.foot) + '</div></div>';
    }).join('') + '<div class="stat-card" style="grid-column:1/-1"><div class="stat-label">主题分布</div>' +
      '<div style="margin-top:6px">' + topics + '</div></div>';

    // 24 小时活跃度
    var hourly = p.hourly || [];
    var items = hourly.map(function (v, h) {
      return { label: pad2(h) + ':00', value: v, title: h + ' 点 ' + v + ' 条' };
    });
    renderSingleBars($('#pf-hours'), $('#pf-hours-axis'), items, ' 条');
    var peak = $('#pf-peak');
    if (peak) {
      var ph = (p.peak_hours || []).map(function (h) { return h + '点'; }).join('、');
      peak.textContent = ph ? ('高峰 ' + ph) : '—';
    }

    // 核心成员
    var members = $('#pf-members');
    if (members) {
      var list = p.core_members || [];
      members.innerHTML = list.length ? list.map(function (m, i) {
        return '<div class="kv-row"><span class="kv-key">#' + (i + 1) + ' ' + esc(m.nickname || m.user_id) +
          '</span><span class="kv-val">' + fmtNum(m.count) + ' 条</span></div>';
      }).join('') : '<div class="empty">还没有数据</div>';
    }

    // 黑话
    var slang = $('#pf-slang');
    if (slang) {
      var pairs = Object.keys(p.slang || {});
      slang.innerHTML = pairs.length
        ? '<div style="padding:4px 0">' + pairs.map(function (k) {
            return '<div class="kv-row"><span class="kv-key">' + esc(k) +
              '</span><span class="kv-val">' + esc(p.slang[k]) + '</span></div>';
          }).join('') + '</div>'
        : '<div class="empty" style="padding:10px 0">还没学到群内特有的说法</div>';
    }
  }

  function renderPolicy(pol, gid) {
    var box = $('#pf-policy');
    if (!box) return;
    var state = $('#pf-policy-state');
    if (state) {
      state.textContent = pol.cold_start
        ? ('冷启动中 ' + pol.samples + '/100 —— 暂不调整参数')
        : ('已学习 ' + pol.samples + ' 次发言');
    }
    var rows = [
      ['发言门槛修正', (pol.threshold_delta >= 0 ? '+' : '') + pol.threshold_delta.toFixed(3),
       '正数=更谨慎，负数=更敢说'],
      ['主动性', pol.initiative.toFixed(3), '主动插话的意愿'],
      ['表情包使用率', fmtPct(pol.sticker_rate), '回复中带自定义表情的比例'],
      ['QQ 表情使用率', fmtPct(pol.emoji_rate), '回复中带小黄脸的比例'],
      ['引用倾向', fmtPct(pol.quote_p), 'Thompson 后验：这个群吃不吃"引用"'],
      ['@ 倾向', fmtPct(pol.at_p), 'Thompson 后验：这个群吃不吃 @'],
      ['样本数', fmtNum(pol.samples), '前 100 次不参与学习']
    ];
    box.innerHTML = rows.map(function (r) {
      return '<div class="kv-row"><span class="kv-key">' + esc(r[0]) +
        '</span><span class="kv-val" title="' + esc(r[2]) + '">' + esc(r[1]) + '</span></div>';
    }).join('');
  }

  function renderReward(points) {
    var items = (points || []).map(function (p) {
      return {
        label: fmtTime(p.ts).slice(0, 5),
        value: Math.max(0, p.reward),
        title: fmtTime(p.ts).slice(0, 5) + ' 平均奖励 ' + p.reward + '（' + p.count + ' 次）'
      };
    });
    renderSingleBars($('#pf-reward'), $('#pf-reward-axis'), items);
    var sum = $('#pf-reward-sum');
    if (sum) {
      if (!items.length) { sum.textContent = '还没有反馈数据'; }
      else {
        var total = points.reduce(function (a, p) { return a + p.count; }, 0);
        var avg = points.reduce(function (a, p) { return a + p.reward * p.count; }, 0) / Math.max(1, total);
        sum.textContent = total + ' 个信号 · 平均 ' + avg.toFixed(2);
      }
    }
  }

  function bindProfile() {
    var restart = $('#ob-restart');
    if (restart) {
      restart.addEventListener('click', function () {
        if (!profileGroup) { toast('先选一个群', 'warn'); return; }
        if (!confirm('重新读一遍这个群的聊天记录并分析？已经问过的问题不会重复问。')) return;
        setBusy(restart, true, '分析中…');
        api('/api/onboard/restart', { method: 'POST', body: { group_id: Number(profileGroup) } })
          .then(function (r) {
            setBusy(restart, false, null, '重新分析');
            toast((r && r.reason) || '已开始重新分析', 'success');
            loadProfileFor(profileGroup);
          })
          .catch(function (e) {
            setBusy(restart, false, null, '重新分析');
            reportError(e, '重新分析失败');
          });
      });
    }

    var sel = $('#pf-group');
    if (sel) {
      sel.addEventListener('change', function () {
        profileGroup = sel.value;
        if (profileGroup) loadProfileFor(profileGroup);
      });
    }
    var refresh = $('#pf-refresh');
    if (refresh) {
      refresh.addEventListener('click', function () {
        setBusy(refresh, true, '刷新中…');
        loadProfile().then(function () { setBusy(refresh, false, null, '刷新'); })
          .catch(function () { setBusy(refresh, false, null, '刷新'); });
      });
    }
    var reset = $('#pf-reset');
    if (reset) {
      reset.addEventListener('click', function () {
        if (!profileGroup) { toast('请先选择一个群', 'warn'); return; }
        setBusy(reset, true, '重置中…');
        api('/api/groups/' + encodeURIComponent(profileGroup) + '/policy/reset', { method: 'POST' })
          .then(function (pol) {
            setBusy(reset, false, null, '重置策略');
            toast('该群的策略已回滚到出厂设置', 'success');
            renderPolicy(pol, profileGroup);
          })
          .catch(function (e) { setBusy(reset, false, null, '重置策略'); reportError(e, '重置失败'); });
      });
    }
  }

  /* ==========================================================
   * 10. 表情库
   * ========================================================== */

  function loadStickers() {
    var box = $('#sk-include-blocked');
    var url = '/api/stickers?limit=300' + (box && box.checked ? '&include_blocked=1' : '');
    return Promise.all([api(url), api('/api/stickers/stats'), api('/api/faces')])
      .then(function (res) {
        renderStickerGrid((res[0] && res[0].stickers) || []);
        renderStickerStats(res[1] || {});
        renderFaces(res[2] || {});
      })
      .catch(function (e) { reportError(e, '表情库加载失败'); });
  }

  function renderStickerStats(st) {
    var grid = $('#sk-stats');
    if (!grid) return;
    grid.innerHTML = [
      { label: '已收藏', value: fmtNum(st.stickers), foot: '可以拿来发的' },
      { label: '见过', value: fmtNum(st.stickers_seen), foot: '含没收的，用来判断通用性' },
      { label: '已标注', value: fmtNum(st.stickers_tagged), foot: '从群友用法里学出了情绪' },
      { label: '使用次数', value: fmtNum(st.sticker_uses), foot: '发出去多少次' },
      { label: '待标注观察', value: fmtNum(st.sticker_obs_pending), foot: '攒够样本就会自动打标' },
      { label: '图片缓存', value: fmtNum(st.media_cached), foot: '命中 ' + fmtNum(st.media_cache_hits) + ' 次' }
    ].map(function (c) {
      return '<div class="stat-card"><div class="stat-label">' + esc(c.label) + '</div>' +
        '<div class="stat-value">' + esc(c.value) + '</div>' +
        '<div class="stat-foot">' + esc(c.foot) + '</div></div>';
    }).join('');

    var rules = $('#sk-rules');
    if (rules) {
      var reasons = st.collect_reject_reasons || {};
      var keys = Object.keys(reasons);
      rules.innerHTML = keys.length ? keys.map(function (k) {
        return '<div class="kv-row"><span class="kv-key">' + esc(k) +
          '</span><span class="kv-val">' + fmtNum(reasons[k]) + ' 次</span></div>';
      }).join('') : '<div class="empty">还没有需要拒绝的表情</div>';
    }

    var moods = $('#sk-moods');
    if (moods) {
      var list = st.sticker_moods || [];
      var cnt = $('#sk-mood-count');
      if (cnt) cnt.textContent = list.length ? (list.length + ' 种') : '还没有';
      moods.innerHTML = list.length
        ? list.map(function (m) { return '<span class="mood-chip">' + esc(m) + '</span>'; }).join('')
        : '<span class="muted small">库里有带标签的表情之后，这里会列出可用的情绪</span>';
    }
  }

  function renderStickerGrid(items) {
    var grid = $('#sk-grid');
    var empty = $('#sk-empty');
    if (!grid) return;
    if (!items.length) {
      grid.innerHTML = '';
      if (empty) empty.classList.remove('hidden');
      return;
    }
    if (empty) empty.classList.add('hidden');

    grid.innerHTML = items.map(function (s) {
      var tags = (s.mood_tags || []).map(function (t) {
        return '<span class="tag">' + esc(t) + '</span>';
      }).join('') || '<span class="tag tag-muted">未标注</span>';
      return '<div class="sticker-card' + (s.blocked ? ' is-blocked' : '') + '">' +
        '<img class="sticker-thumb" loading="lazy" src="/api/stickers/' + esc(s.id) + '/file" alt="表情">' +
        '<div>' + tags + '</div>' +
        (s.desc ? '<div class="muted small">' + esc(s.desc) + '</div>' : '') +
        '<div class="sticker-meta"><span>用 ' + fmtNum(s.use_count) + '</span>' +
        '<span>赢 ' + fmtNum(s.win_count) + '</span><span>见 ' + fmtNum(s.seen_count) + '</span></div>' +
        (s.reason ? '<div class="muted small" title="收藏理由">' + esc(s.reason) + '</div>' : '') +
        '<div class="sticker-actions">' +
        '<button class="btn btn-secondary btn-sm" data-sticker-block="' + esc(s.id) + '" data-blocked="' + (s.blocked ? 1 : 0) + '" type="button">' +
        (s.blocked ? '解禁' : '封禁') + '</button>' +
        '<button class="btn btn-secondary btn-sm" data-sticker-del="' + esc(s.id) + '" type="button">删除</button>' +
        '</div></div>';
    }).join('');
  }

  function renderFaces(data) {
    var box = $('#sk-faces');
    if (!box) return;
    var core = (data.core || []);
    var cnt = $('#sk-face-count');
    if (cnt && data.stats) {
      cnt.textContent = '共 ' + data.stats.sysface + ' 个系统表情 · 常用 ' + data.stats.core + ' 个已注入 prompt';
    }
    box.innerHTML = core.map(function (f) {
      return '<span class="face-chip" title="id ' + esc(f.id) + '">' + esc(f.name) + '</span>';
    }).join('');
  }

  function bindStickers() {
    var box = $('#sk-include-blocked');
    if (box) box.addEventListener('change', function () { loadStickers(); });

    var refresh = $('#sk-refresh');
    if (refresh) {
      refresh.addEventListener('click', function () {
        setBusy(refresh, true, '刷新中…');
        loadStickers().then(function () { setBusy(refresh, false, null, '刷新'); })
          .catch(function () { setBusy(refresh, false, null, '刷新'); });
      });
    }

    document.addEventListener('click', function (ev) {
      var btn = ev.target.closest ? ev.target.closest('[data-sticker-block],[data-sticker-del]') : null;
      if (!btn) return;

      var blockId = btn.getAttribute('data-sticker-block');
      if (blockId) {
        var nowBlocked = btn.getAttribute('data-blocked') === '1';
        setBusy(btn, true, '…');
        api('/api/stickers/' + encodeURIComponent(blockId) + '/block', {
          method: 'POST', body: { blocked: !nowBlocked }
        }).then(function () {
          toast(nowBlocked ? '已解禁' : '已封禁，不会再发这张', 'success');
          return loadStickers();
        }).catch(function (e) { reportError(e, '操作失败'); });
        return;
      }

      var delId = btn.getAttribute('data-sticker-del');
      if (delId) {
        setBusy(btn, true, '…');
        api('/api/stickers/' + encodeURIComponent(delId), { method: 'DELETE' })
          .then(function () { toast('已删除', 'success'); return loadStickers(); })
          .catch(function (e) { reportError(e, '删除失败'); });
      }
    });
  }


  /* ==========================================================
   * B 站扫码登录
   * ========================================================== */

  var biliQrTimer = null;

  /** 拉当前登录态。Cookie 掉了就把提示变成显眼的告警。 */
  function loadBiliStatus() {
    var st = $('#bili-login-state');
    if (!st) return Promise.resolve();
    return api('/api/bilibili/status').then(function (d) {
      d = d || {};
      st.className = 'bili-login-state ' + (d.ok ? 'ok' : (d.need_login ? 'bad' : 'warn'));
      st.textContent = d.message || '未知';
      var btn = $('#bili-login-btn');
      if (btn) btn.textContent = d.need_login ? '扫码登录（Cookie 已失效）' : '扫码登录';
      // Cookie 掉了就自动把二维码亮出来 —— 用户不用自己点
      if (d.need_login && !biliQrTimer) biliQrStart();
    }).catch(function () {
      st.className = 'bili-login-state warn';
      st.textContent = '拉取登录态失败';
    });
  }

  function biliQrStart() {
    var box = $('#bili-qr-box'), img = $('#bili-qr-img'), msg = $('#bili-qr-msg');
    var btn = $('#bili-login-btn');
    setBusy(btn, true, '获取中…');
    return api('/api/bilibili/login/qrcode', { method: 'POST' })
      .then(function (d) {
        setBusy(btn, false, null, '重新获取');
        d = d || {};
        if (!d.qr) {
          if (msg) msg.textContent = d.message || '二维码获取失败';
          if (box) box.classList.remove('hidden');
          return;
        }
        if (img) img.src = d.qr;
        if (box) box.classList.remove('hidden');
        if (msg) msg.textContent = d.state_text || '等待扫码…';
        biliQrPoll();
      })
      .catch(function (e) {
        setBusy(btn, false, null, '扫码登录');
        reportError(e, '获取二维码失败');
      });
  }

  /** 每 2 秒问一次。服务端在成功那一次会当场把 Cookie 装上。 */
  function biliQrPoll() {
    if (biliQrTimer) clearInterval(biliQrTimer);
    biliQrTimer = setInterval(function () {
      api('/api/bilibili/login/status').then(function (d) {
        d = d || {};
        var msg = $('#bili-qr-msg');
        if (msg) msg.textContent = d.state_text || '';
        if (d.state === 'confirmed') {
          clearInterval(biliQrTimer); biliQrTimer = null;
          toast('B站登录成功' + (d.uname ? '：' + d.uname : ''), 'success');
          var box = $('#bili-qr-box');
          if (box) box.classList.add('hidden');
          loadBiliStatus();
          if (d.settings && typeof applySettings === 'function') applySettings(d.settings);
        } else if (d.state === 'expired' || d.state === 'error') {
          clearInterval(biliQrTimer); biliQrTimer = null;
          setBusy($('#bili-login-btn'), false, null, '重新获取');
        }
      }).catch(function () { /* 网络抖一下，继续轮询 */ });
    }, 2000);
  }


  /* ==========================================================
   * 模型适配
   * ========================================================== */

  var modelProfiles = null;

  function loadModelAdvice() {
    var box = $('#model-advice-body');
    if (!box) return Promise.resolve();
    return api('/api/models/profiles').then(function (d) {
      modelProfiles = d || {};
      renderModelAdvice(d);
    }).catch(function () {
      box.textContent = '拉取模型画像失败';
    });
  }

  function renderModelAdvice(d) {
    var box = $('#model-advice-body');
    if (!box) return;
    d = d || {};
    var cur = d.current || {};
    var warns = cur.warnings || [];
    var html = '';
    html += '<div class="model-row">回复 <b>' + esc(cur.reply_model || '-') + '</b> · '
          + esc(cur.reply_profile || '') + '</div>';
    html += '<div class="model-row">分析 <b>' + esc(cur.analysis_model || '-') + '</b> · '
          + esc(cur.analysis_profile || '') + '</div>';
    warns.forEach(function (w) {
      html += '<div class="model-advice-warn">⚠ ' + esc(w) + '</div>';
    });
    if (!warns.length) {
      html += '<div class="model-advice-ok">✓ 当前模型搭配没有已知问题</div>';
    }
    box.innerHTML = html;
  }


  /** 按当前输入实时估算「在」「笑死我了」这类短句要多久发出。 */
  function renderTypingPreview() {
    var box = $('#typing-preview');
    if (!box) return;
    var lo = numOf('set-typing-min') || 180;
    var hi = numOf('set-typing-max') || 420;
    var floor = numOf('set-typing-floor');
    if (isNaN(floor)) floor = 1.0;
    var rlo = numOf('set-reaction-min');
    if (isNaN(rlo)) rlo = 0.6;
    var rhi = numOf('set-reaction-max');
    if (isNaN(rhi)) rhi = 2.0;
    var avg = (lo + hi) / 2 / 1000;
    var reaction = (rlo + rhi) / 2;

    // 打字时间（不含反应）
    function typeOnly(units) {
      return Math.max(floor * 1.15, units * avg);
    }
    // 一次回复 = 一次反应 + 每条气泡各一次打字时间
    function reply(unitsList) {
      var t = reaction;
      unitsList.forEach(function (u) { t += typeOnly(u); });
      return t;
    }
    box.textContent = '「在」约 ' + reply([1]).toFixed(1) + 's · 「笑死我了」约 '
      + reply([4]).toFixed(1) + 's · 三条各 3 字约 '
      + reply([3, 3, 3]).toFixed(0) + 's';
  }

  function bindTypingPreview() {
    ['set-typing-min', 'set-typing-max', 'set-typing-floor',
     'set-reaction-min', 'set-reaction-max'].forEach(function (id) {
      var el = $('#' + id);
      if (el) el.addEventListener('input', renderTypingPreview);
    });
    renderTypingPreview();
  }

  function bindModelAdvice() {
    var btn = $('#model-apply');
    if (btn) {
      btn.addEventListener('click', function () {
        setBusy(btn, true, '应用中…');
        api('/api/models/apply', { method: 'POST', body: { target: 'both' } })
          .then(function (r) {
            setBusy(btn, false, null, '应用推荐配置');
            r = r || {};
            if (r.settings && typeof applySettings === 'function') applySettings(r.settings);
            var applied = r.applied || {};
            toast('已应用推荐配置：' + JSON.stringify(applied), 'success');
            loadModelAdvice();
          })
          .catch(function (e) {
            setBusy(btn, false, null, '应用推荐配置');
            reportError(e, '应用失败');
          });
      });
    }
    loadModelAdvice();
  }

  function bindBiliLogin() {
    var btn = $('#bili-login-btn');
    if (btn) btn.addEventListener('click', function () { biliQrStart(); });
    loadBiliStatus();
  }

  function bindSearchTest() {
    var btn = $('#search-test-btn');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var input = $('#search-test-q');
      var out = $('#search-test-out');
      var q = (input && input.value.trim()) || '今天的新闻';
      setBusy(btn, true, '搜索中…');
      var t0 = Date.now();
      api('/api/settings/search/test', { method: 'POST', body: { query: q } })
        .then(function (r) {
          setBusy(btn, false, null, '测试');
          r = r || {};
          var ms = r.latency_ms != null ? r.latency_ms : (Date.now() - t0);
          if (!r.ok) {
            if (out) { out.textContent = (r.error || '搜索失败') + '（' + ms + ' ms）'; out.style.color = 'var(--danger)'; }
            toast('搜索测试失败', 'error');
            return;
          }
          var lines = ['[' + r.provider + '] ' + r.results.length + ' 条结果 · ' + ms + ' ms'];
          r.results.forEach(function (x, i) {
            lines.push((i + 1) + '. ' + x.title);
            if (x.snippet) lines.push('   ' + x.snippet.slice(0, 90));
            if (x.url) lines.push('   ' + x.url);
          });
          if (out) { out.textContent = lines.join('\n'); out.style.color = 'var(--success)'; }
          toast('搜索正常', 'success');
        })
        .catch(function (e) {
          setBusy(btn, false, null, '测试');
          var m = (e && e.message) || '搜索请求失败';
          if (out) { out.textContent = m; out.style.color = 'var(--danger)'; }
          reportError(e, '搜索测试失败');
        });
    });
  }

  function bindDebugBanner() {
    var btn = $('#debug-off');
    if (!btn) return;
    btn.addEventListener('click', function () {
      setBusy(btn, true, '关闭中…');
      api('/api/control/debug', { method: 'POST', body: { enabled: false } })
        .then(function () {
          setBusy(btn, false, null, '立即关闭');
          toast('测试模式已关闭', 'success');
          refreshDashboardStatus();
        })
        .catch(function (e) {
          setBusy(btn, false, null, '立即关闭');
          reportError(e, '关闭失败');
        });
    });
  }

  /* 试生成：跑一次真实生成链路，但不发送 */
  function bindDryRun() {
    var btn = $('#dry-run');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var input = $('#dry-text');
      var out = $('#dry-out');
      var text = input ? input.value.trim() : '';
      if (!text) {
        if (out) { out.textContent = '先输入一条模拟消息。'; out.style.color = 'var(--danger)'; }
        return;
      }
      var gid = (groupsCache && groupsCache.length) ? groupsCache[0].group_id : 0;
      if (!gid) {
        if (out) { out.textContent = '还没有已发现的群，先让机器人在某个群里收到一条消息。'; out.style.color = 'var(--danger)'; }
        return;
      }
      var force = !!($('#dry-force') && $('#dry-force').checked);
      setBusy(btn, true, '生成中…');
      var t0 = Date.now();
      api('/api/control/test', { method: 'POST', body: { group_id: gid, text: text, force: force } })
        .then(function (r) {
          setBusy(btn, false, null, '试生成');
          r = r || {};
          var ms = Date.now() - t0;
          if (r.error) {
            if (out) { out.textContent = r.error; out.style.color = 'var(--danger)'; }
            toast('生成失败', 'error');
            return;
          }
          var head = r.should_reply
            ? ('决定层：发言（' + r.reason + '，分数 ' + r.priority + '）')
            : ('决定层：不发言（' + r.reason + '，分数 ' + r.priority + '）');
          var body = (r.draft && r.draft.length)
            ? ('\n模型输出：' + r.draft.map(function (x) { return '「' + x + '」'; }).join(' '))
            : '\n模型没有输出可发内容';
          if (out) {
            out.textContent = head + body + '\n耗时 ' + ms + ' ms';
            out.style.color = r.should_reply ? 'var(--success)' : 'var(--text-faint)';
          }
        })
        .catch(function (e) {
          setBusy(btn, false, null, '试生成');
          reportError(e, '试生成失败');
        });
    });
  }


  /* ==========================================================
   * 11. 记忆
   * ========================================================== */

  var memGroupsCache = [];

  var MEMORY_TYPE = {
    fact: '事实', relation: '关系', culture: '群文化', slang: '黑话',
    event: '事件', preference: '偏好', topic: '主题', emotion: '情绪',
    self: '自我'
  };

  /** 半衰期 -> 人话。区分"近乎永久"和"只有几天"，一眼看出会不会忘。 */
  function fmtTtl(days) {
    if (!days) return '—';
    if (days >= 3650) return '近乎永久';
    if (days >= 365) return Math.round(days / 365) + ' 年';
    return days + ' 天';
  }

  function totalOf(byType) {
    var n = 0;
    Object.keys(byType || {}).forEach(function (k) { n += Number(byType[k]) || 0; });
    return n;
  }

  function memoryBadgeClass(days) {
    if (!days) return '';
    if (days >= 3650) return 'long';
    if (days <= 30) return 'short';
    return '';
  }

  function loadMemories() {
    var sel = $('#mem-group');
    var invalid = $('#mem-include-invalid');
    var params = [];
    if (sel && sel.value) {
      params.push('scope=group');
      params.push('scope_id=' + encodeURIComponent(sel.value));
    }
    if (!(invalid && invalid.checked)) params.push('include_invalid=0');
    params.push('limit=400');

    return api('/api/memories?' + params.join('&'))
      .then(function (data) {
        renderMemoryStats(data.summary || {});
        renderMemoryFilters(data.summary || {});
        renderMemoryList((data && data.memories) || []);
      })
      .catch(function (e) { reportError(e, '记忆加载失败'); });
  }

  function renderMemoryStats(sum) {
    var grid = $('#mem-stats');
    if (!grid) return;
    var types = sum.by_type || {};
    grid.innerHTML = [
      { label: '记忆总数', value: fmtNum(sum.total), foot: '所有群加起来' },
      { label: '有效条数', value: fmtNum(totalOf(types)), foot: '当前筛选范围内' },
      { label: '已失效', value: fmtNum(sum.invalidated), foot: '被新信息推翻，保留作历史' },
      { label: '单群上限', value: fmtNum(sum.capacity_per_group), foot: '超了淘汰最没用的' }
    ].map(function (c) {
      return '<div class="stat-card"><div class="stat-label">' + esc(c.label) + '</div>' +
        '<div class="stat-value">' + esc(c.value) + '</div>' +
        '<div class="stat-foot">' + esc(c.foot) + '</div></div>';
    }).join('');

    var box = $('#mem-types');
    if (box) {
      var ttl = sum.ttl_by_type || {};
      var keys = Object.keys(types);
      box.innerHTML = keys.length ? keys.map(function (k) {
        var cls = memoryBadgeClass(ttl[k]);
        return '<div class="kv-row"><span class="kv-key">' +
          '<span class="mem-badge ' + cls + '">' + esc(MEMORY_TYPE[k] || k) + '</span></span>' +
          '<span class="kv-val">' + fmtNum(types[k]) + ' 条 · 记 ' + esc(fmtTtl(ttl[k])) + '</span></div>';
      }).join('') : '<div class="empty">还没有记忆</div>';
    }

    var gbox = $('#mem-groups');
    var cap = $('#mem-cap');
    var groups = sum.groups || [];
    if (cap) cap.textContent = groups.length ? (groups.length + ' 个群有记忆') : '—';
    if (gbox) {
      gbox.innerHTML = groups.length ? groups.map(function (g) {
        var right = g.active > 0
          ? (fmtNum(g.active) + ' 条' +
             (g.invalidated ? ' <span class="muted small">(+' + g.invalidated + ' 失效)</span>' : ''))
          : '<span class="muted small">暂无</span>';
        return '<div class="kv-row"><span class="kv-key">' +
          (g.enabled ? '' : '<span class="muted small">·</span> ') + esc(g.name) + '</span>' +
          '<span class="kv-val">' + right + '</span></div>';
      }).join('') : '<div class="empty">还没有发现任何群</div>';
    }
  }

  function renderMemoryFilters(sum) {
    memGroupsCache = sum.groups || [];
    var sel = $('#mem-group');
    if (!sel) return;
    var prev = sel.value;
    var html = '<option value="">全部群</option>';
    (sum.groups || []).forEach(function (g) {
      var label = g.active > 0 ? (g.active + ' 条') : '暂无记忆';
      if (g.invalidated) label += ' · ' + g.invalidated + ' 失效';
      html += '<option value="' + esc(g.group_id) + '">' +
        esc((g.enabled ? '' : '· ') + g.name + '（' + label + '）') + '</option>';
    });
    sel.innerHTML = html;
    if (prev) sel.value = prev;
  }

  /** 空态文案要贴合上下文：选了某个群就别说"还没有记忆"这么笼统。 */
  function memoryEmptyHint() {
    var sel = $('#mem-group');
    var gid = sel && sel.value;
    if (!gid) {
      return '还没有任何记忆。机器人在每个群里每积累 40 条消息抽取一次，' +
             '只记「明天还成立、以后用得上、上下文推不出来」的内容。';
    }
    var g = null;
    (memGroupsCache || []).forEach(function (x) {
      if (String(x.group_id) === String(gid)) g = x;
    });
    if (!g) return '这个群还没有记忆。';
    var parts = ['「' + esc(g.name) + '」还没有记忆。'];
    parts.push('该群累计 ' + fmtNum(g.msg_count) + ' 条消息 —— 每 40 条触发一次抽取' +
               (g.msg_count >= 40 ? '，已经够一轮了，等下次触发即可。'
                                  : '，还需要 ' + (40 - g.msg_count) + ' 条。'));
    if (!g.enabled) {
      parts.push('<b>注意：这个群当前是关闭状态</b>，只记录不发言，也不会抽取记忆。');
    }
    return parts.join(' ');
  }

  function renderMemoryList(items) {
    var box = $('#mem-list');
    var empty = $('#mem-empty');
    if (!box) return;
    if (!items.length) {
      box.innerHTML = '';
      if (empty) {
        empty.classList.remove('hidden');
        empty.innerHTML = memoryEmptyHint();
      }
      return;
    }
    if (empty) empty.classList.add('hidden');

    var ttl = {};
    box.innerHTML = items.map(function (m) {
      var invalid = !!m.invalid_at;
      var cls = invalid ? 'invalid' : memoryBadgeClass(m.ttl_days);
      var facts = [
        '置信 ' + (m.confidence != null ? Number(m.confidence).toFixed(2) : '—'),
        '记 ' + fmtTtl(m.ttl_days),
        '建于 ' + fmtTime(m.created_ts).slice(0, 5)
      ];
      if (m.last_hit_ts) facts.push('想起来过');
      if (invalid) facts.push('已于 ' + fmtTime(m.invalid_at).slice(0, 5) + ' 被推翻');
      if (m.source && m.source !== 'observed') facts.push('来源 ' + m.source);

      return '<div class="mem-item' + (invalid ? ' is-invalid' : '') + '">' +
        '<div class="mem-head">' +
        '<span class="mem-badge ' + cls + '">' + esc(MEMORY_TYPE[m.type] || m.type || '?') + '</span>' +
        '<div class="mem-body">' + esc(m.content) + '</div>' +
        '<button class="btn btn-secondary btn-sm" data-mem-del="' + esc(m.id) + '" type="button">删除</button>' +
        '</div>' +
        '<div class="mem-meta">' + facts.map(esc).join('<span></span>') + '</div>' +
        '</div>';
    }).join('');
  }

  function bindMemories() {
    var sel = $('#mem-group');
    if (sel) sel.addEventListener('change', function () { loadMemories(); });

    var inc = $('#mem-include-invalid');
    if (inc) inc.addEventListener('change', function () { loadMemories(); });

    var refresh = $('#mem-refresh');
    if (refresh) {
      refresh.addEventListener('click', function () {
        setBusy(refresh, true, '刷新中…');
        loadMemories().then(function () { setBusy(refresh, false, null, '刷新'); })
          .catch(function () { setBusy(refresh, false, null, '刷新'); });
      });
    }

    document.addEventListener('click', function (ev) {
      var btn = ev.target.closest ? ev.target.closest('[data-mem-del]') : null;
      if (!btn) return;
      var id = btn.getAttribute('data-mem-del');
      setBusy(btn, true, '…');
      api('/api/memories/' + encodeURIComponent(id), { method: 'DELETE' })
        .then(function () { toast('已删除该条记忆', 'success'); return loadMemories(); })
        .catch(function (e) { setBusy(btn, false, null, '删除'); reportError(e, '删除失败'); });
    });
  }

  function boot() {
    if (booted) return Promise.resolve();
    return api('/api/me').then(function (me) {
      if (!me || me.authenticated !== true) {
        showLogin();
        return;
      }
      booted = true;
      appVersion = me.version || '—';
      var ver = $('#side-version');
      if (ver) ver.textContent = 'v' + appVersion;
      showApp();
      if (!uiBound) {
        uiBound = true;
        bindNav();
      }
      showView(parseHash());
      // 常驻状态轮询：即使直接落在非概览页，连接胶囊也要正确
      startDashboardTimer();
      refreshDashboardStatus();
      // 群列表供多个视图复用，登录后预热一次
      api('/api/groups').then(function (list) {
        groupsCache = Array.isArray(list) ? list : [];
        fillGroupSelect($('#tl-group'), '请选择群', groupsCache);
        fillGroupSelect($('#dec-group'), '全部群', groupsCache, true);
      }).catch(function () { /* 预热失败不打扰，进入对应视图会重试 */ });
    }).catch(function (e) {
      if (e && e.status === 401) return; // api() 已切到登录页
      showLogin();
      setLoginError((e && e.message) || '无法连接服务端。');
    });
  }

  function init() {
    toastLayer = $('#toast-layer');
    bindLogin();
    bindLogout();
    bindGroups();
    bindTimeline();
    bindDecisions();
    bindPersona();
    bindPrompts();
    bindProfile();
    bindMemories();
    bindStickers();
    bindSettings();
    bindDryRun();
    bindDebugBanner();
    bindSearchTest();
    bindBiliLogin();
    bindModelAdvice();
    bindTypingPreview();
    bindLogs();

    window.addEventListener('hashchange', function () {
      var name = parseHash();
      if (name === currentView) return;   // 自己写 hash 触发的事件，忽略
      showView(name);
    });

    window.addEventListener('beforeunload', function () {
      stopDashboardTimer();
      if (logSource) { try { logSource.close(); } catch (e) { /* ignore */ } }
    });

    boot();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
