"""不带密钥、不生成内容的连通检查；TLS校验保持开启。"""
from pathlib import Path
import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler, getproxies, proxy_bypass
from urllib.parse import urlsplit

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from notice_pipeline.client import network_error_code, NETWORK_MESSAGES
from notice_pipeline.source import load_config, run_path
from notice_pipeline.storage import now, write_json


def probe(api, *, direct=False):
    url=api['base_url'].rstrip('/')+'/models'
    bypass=direct or proxy_bypass(urlsplit(url).hostname)
    result=dict(checked_at=now(), route='direct' if bypass else 'system',
                proxy_bypassed_for_api=bypass,
                proxy_configured=bool(getproxies()), sent_api_key=False,
                generation_requests=0, ready=False)
    opener=build_opener(ProxyHandler({})) if direct else build_opener()
    try:
        with opener.open(Request(url,method='GET'),timeout=12) as response:
            result['http_status']=response.status
    except HTTPError as exc:
        result['http_status']=exc.code
        exc.close()
    except (URLError, OSError, TimeoutError) as exc:
        code=network_error_code(exc)
        result.update(error_code=code,error=NETWORK_MESSAGES[code])
    # 未授权的401是预期回应，能证明TLS/HTTP链路通；不验证密钥或余额。
    result['ready']=result.get('http_status') in {200,401}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=P/'configs/supervision_batch1.toml')
    parser.add_argument('--direct',action='store_true',help='仅诊断直连；不修改系统代理或付费请求路由')
    args=parser.parse_args()
    cfg=load_config(args.config)
    result=probe(cfg['api'],direct=args.direct)
    write_json(run_path(cfg)/('connection_check_direct.json' if args.direct else 'connection_check.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['ready']:
        print('连通检查未通过，先修复网络；本检查没有发送密钥或生成请求。',file=sys.stderr)
    return 0 if result['ready'] else 2


if __name__=='__main__':raise SystemExit(main())
