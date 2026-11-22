const { chromium } = require('playwright-core');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const SOURCE = '/home/kingdove/QQAIRobot';
const ROOT = '/tmp/betterai_capture_root';
const OUT = path.join(SOURCE, 'docs/screenshots');
const DB = '/tmp/betterai_public_demo.db';
const PORT = 8097;
const URL = 'http://127.0.0.1:' + PORT;

const VIEWS = [
  ['01-dashboard.png','dashboard',1500,560],
  ['02-groups.png','groups',1500,1150],
  ['03-timeline.png','timeline',1500,1150],
  ['04-decisions.png','decisions',1500,1150],
  ['05-persona.png','persona',1500,1150],
  ['06-settings.png','settings',1500,1150],
  ['07-logs.png','logs',1500,1100],
  ['08-group-profile.png','profile',1500,1250],
  ['09-stickers.png','stickers',1500,1250],
  ['10-icon.png','icon',620,290],
  ['11-memory.png','memory',1500,1150],
  ['12-persona-scope.png','persona',1500,1250],
  ['13-prompt.png','prompt',1500,1500],
];

function run(cmd,args,env,opts={}) {
  return new Promise((resolve,reject)=>{
    const p=spawn(cmd,args,{cwd:ROOT,env:{...process.env,...env},stdio:opts.quiet?'ignore':'inherit'});
    p.on('exit',(code)=>code===0?resolve():reject(new Error(cmd+' '+args.join(' ')+' exit '+code)));
  });
}
const sleep=(ms)=>new Promise(r=>setTimeout(r,ms));

(async()=>{
  // Clean-room copy: tracked source only; no .env, data, or old screenshots.
  fs.rmSync(ROOT,{recursive:true,force:true});
  fs.mkdirSync(ROOT,{recursive:true});
  const dirs=['tools','qqbot','docs/screenshots'];
  const files=['config.toml','persona.yaml','safety.txt','requirements.txt'];
  fs.mkdirSync(path.join(ROOT,'docs'),{recursive:true});
  for(const d of dirs) fs.cpSync(path.join(SOURCE,d),path.join(ROOT,d),{recursive:true});
  for(const file of files) fs.copyFileSync(path.join(SOURCE,file),path.join(ROOT,file));
  await run(path.join(SOURCE,'.venv/bin/python'),['-m','tools.seed_demo'],{QQBOT_DB_PATH:DB,PYTHONPATH:ROOT}, {quiet:true});
  console.log('seeded demo DB');

  const server=spawn(path.join(SOURCE,'.venv/bin/python'),['-m','qqbot','web'],{
    cwd:ROOT,
    env:{
      ...process.env,
      QQBOT_DB_PATH:DB,
      QQBOT_WEB_PORT:String(PORT),
      WEB_HOST:'127.0.0.1',
      WEB_TOKEN:'',
      ONEBOT_WS_URL:'ws://127.0.0.1:9',
      LOG_LEVEL:'INFO',
      PYTHONPATH:ROOT,
    },
    stdio:['ignore','pipe','pipe'],
  });
  server.stdout.on('data',()=>{}); server.stderr.on('data',()=>{});
  try {
    let ok=false;
    for(let i=0;i<80;i++){
      try { const r=await fetch(URL+'/api/me'); if(r.ok||r.status===401){ok=true;break;} }catch(e){}
      await sleep(250);
    }
    if(!ok) throw new Error('server did not start');
    console.log('web server ready');

    const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
    const ctx=await browser.newContext({viewport:{width:1500,height:1200},deviceScaleFactor:1,locale:'zh-CN'});
    const page=await ctx.newPage();
    // Login screen: token empty means no auth; add one deliberate wait so CSS/fonts settle.
    await page.goto(URL,{waitUntil:'domcontentloaded'});
    await page.waitForTimeout(1200);
    await page.screenshot({path:path.join(OUT,'00-login.png')});
    console.log('00-login');

    for(const [file,view,w,h] of VIEWS){
      if(file==='10-icon.png'){
        const iconPage=await ctx.newPage();
        await iconPage.setViewportSize({width:w,height:h});
        const icon=await fs.promises.readFile(path.join(ROOT,'qqbot/web/static/icon.svg'),'utf8');
        await iconPage.setContent('<body style="margin:0;display:flex;align-items:center;justify-content:center;width:620px;height:290px;background:#fff">'+icon.replace('<svg','<svg width="100" height="100"')+'</body>');
        await iconPage.waitForTimeout(200);
        await iconPage.screenshot({path:path.join(OUT,file)});
        await iconPage.close();
        console.log(file); continue;
      }
      await page.setViewportSize({width:w,height:h});
      await page.goto(URL+'/#/'+view,{waitUntil:'domcontentloaded'});
      await page.waitForTimeout(3000);
      if(view==='profile'){
        // choose the first demo group if a dropdown exists
        const select=await page.$('select');
        if(select) { try{await select.selectOption({index:0});await page.waitForTimeout(800);}catch(e){} }
      }
      await page.screenshot({path:path.join(OUT,file)});
      console.log(file);
    }
    await browser.close();
  } finally {
    server.kill('SIGKILL');
    await sleep(300);
  }
  console.log('done');
})().catch(e=>{console.error(e);process.exit(1)});
