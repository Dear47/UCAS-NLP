import os
import hashlib
import re
from collections import deque
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import asyncio
import aiohttp
from aiohttp import ClientSession
import chardet

class AsyncCorpusCrawler:
    """
    一个异步并发的网络爬虫, 用于爬取指定网站的中文或英文语料
    它会遵守 robots.txt 规则, 进行速率限制, 并对文本进行清洗
    """
    
    def __init__(self, start_urls, output_dir, max_pages=50, delay=1, lang='en', max_concurrent=10):
        """
        初始化爬虫

        :param start_urls: (list) 起始URL列表
        :param output_dir: (str) .txt文件的输出目录
        :param max_pages: (int) 最大爬取页面数
        :param delay: (int) 两次请求之间的延迟(秒)
        :param lang: (str) 语料语言 ('en' 或 'zh'), 用于决定清洗规则
        """
        self.start_urls = start_urls
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.delay = delay
        self.lang = lang 
        self.max_concurrent = max_concurrent
        
        self.urls_to_visit = deque(start_urls)
        self.visited_urls = set()
        self.robot_parsers = {}
        # self.user_agent = 'MyCorpusCrawler/1.0 (+https://github.com/your-repo)'
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'  # 不同的代理可以爬取不同的网站
        self.headers = {'User-Agent': self.user_agent}
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[*] 输出目录: {self.output_dir}")
        print(f"[*] 爬虫模式: {self.lang}")
        print(f"[*] 最大并发数:{self.max_concurrent}")

    def _get_robot_parser(self, url):
        """
        获取robot.txt
        """
        domain = urlparse(url).netloc
        if domain not in self.robot_parsers:
            robots_url = f"{urlparse(url).scheme}://{domain}/robots.txt"
            print(f"[*] 正在获取 robots.txt: {robots_url}")
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robot_parsers[domain] = rp
            except Exception as e:
                print(f"[!] 无法读取 robots.txt: {robots_url} (错误: {e})")
                self.robot_parsers[domain] = None 
        return self.robot_parsers[domain]

    def _can_fetch(self, url):
        """
        检查是否可爬
        """
        rp = self._get_robot_parser(url)
        if rp:
            return rp.can_fetch(self.user_agent, url)
        return True

    def _clean_chinese(self, text):
        """
        清洗中文文本: 只保留中文字符, 其他替换为空格
        """
        # 匹配所有非中文字符 (\u4e00-\u9fff), 将其替换为单个空格
        pattern = re.compile(r'[^\u4e00-\u9fff]')
        cleaned_text = re.sub(pattern, ' ', text)
        
        # 将连续的多个空白(空格、换行等)合并为单个空格
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def _clean_english(self, text):
        """
        清洗英文文本: 只保留26个英文字母, 其他替换为空格
        """
        # 匹配所有非英文字母 (a-z, A-Z)将, 其替换为单个空格
        pattern = re.compile(r'[^a-zA-Z]')
        cleaned_text = re.sub(pattern, ' ', text)
        
        # 将连续的多个空白(空格、换行等)合并为单个空格
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def _extract_text(self, soup):
        """
        从 BeautifulSoup 对象中提取文本, 并根据 self.lang 进行清洗。
        """
        # 移除脚本和样式，避免干扰
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose()
            
        paragraphs = soup.find_all('p')
        text_content = '\n'.join([p.get_text(strip=False) for p in paragraphs if p.get_text(strip=True)])
        
        if self.lang == 'zh':
            cleaned_text = self._clean_chinese(text_content)
        elif self.lang == 'en':
            cleaned_text = self._clean_english(text_content)
        else:
            # 如果语言未指定或不支持，则只做基础的空白清理
            cleaned_text = re.sub(r'\s+', ' ', text_content).strip()
            
        return cleaned_text

    def _save_text(self, url, text):
        """
        将提取的文本保存到 .txt 文件。
        """
        if not text:
            print(f"[*] 页面无有效语料, 跳过保存: {url}")
            return
            
        filename = hashlib.md5(url.encode('utf-8')).hexdigest() + '.txt'
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"[+] 成功保存 (清洗后): {url} -> {filepath}")
        except Exception as e:
            print(f"[!] 保存文件失败 {filepath}: {e}")

    def _find_links(self, current_url, soup):
        """
        在页面上查找新的链接，并将其添加到待访问队列。
        """
        base_domain = urlparse(current_url).netloc
        new_links_found = 0
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            new_url = urljoin(current_url, href)
            new_url = new_url.split('#')[0]
            
            if (new_url.startswith(('http:', 'https:')) and
                urlparse(new_url).netloc == base_domain and
                new_url not in self.visited_urls and
                new_url not in self.urls_to_visit):
                
                self.urls_to_visit.append(new_url)
                new_links_found += 1
        
        if new_links_found > 0:
            print(f"[*] 找到 {new_links_found} 个新链接。队列总数: {len(self.urls_to_visit)}")

    async def _fetch_html(self, session, url):
            try:
                async with session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        print(f"[!] 状态码 {resp.status}: {url}")
                        return None

                    content_type = resp.headers.get('Content-Type', '').lower()
                    if 'text/html' not in content_type:
                        print(f"[!] 非HTML内容类型 ({content_type}): {url}")
                        return None
                    
                    raw_data = await resp.read()

                    encoding = None
                    if 'charset=' in content_type:
                        encoding = content_type.split('charset=')[-1].strip()

                    if not encoding or encoding.lower() not in ('utf-8', 'utf8', 'gbk', 'gb2312', 'gb18030'):
                        detected = chardet.detect(raw_data)
                        encoding = detected['encoding']

                    try:
                        html = raw_data.decode(encoding or 'utf-8', errors='replace')
                    except (UnicodeDecodeError, LookupError):
                        # 如果指定编码无效, 尝试常用中文编码
                        for enc in ['gb18030', 'gbk', 'utf-8']:
                            try:
                                html = raw_data.decode(enc, errors='replace')
                                encoding = enc
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            print(f"[!] 无法解码网页内容: {url}")
                            return None

                    print(f"[DEBUG] 使用编码 {encoding} 解码: {url}")
                    return html
            except Exception as e:
                print(f"[!] 请求失败 {url}: {e}")
                return None

    async def _process_url(self, session, url):
        # if not self._can_fetch(url):
        #     print(f"[-] 被 robots.txt 禁止: {url}")
        #     return []

        print(f"[*] 正在处理: {url}")
        html = await self._fetch_html(session, url)
        if not html:
            return []

        try:
            soup = BeautifulSoup(html, 'html.parser')
            text = self._extract_text(soup)
            self._save_text(url, text)
            new_links = self._find_links(url, soup)
            return new_links
        except Exception as e:
            print(f"[!] 解析失败 {url}: {e}")
            return []

    async def run(self):
        print(f"[*] 开始异步爬取... 最大页面数: {self.max_pages}")

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_process(session, url):
            async with semaphore:
                return await self._process_url(session, url)

        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=15)
        async with ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            processed = 0

            while self.urls_to_visit and processed < self.max_pages:
                url = self.urls_to_visit.popleft()
                if url in self.visited_urls:
                    continue
                self.visited_urls.add(url)

                task = asyncio.create_task(bounded_process(session, url))
                tasks.append(task)
                processed += 1

                # 当达到一批或队列快空时，收集结果
                if len(tasks) >= self.max_concurrent or not self.urls_to_visit:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    tasks.clear()

                    # 收集新链接
                    for result in results:
                        if isinstance(result, list):
                            for link in result:
                                if link not in self.visited_urls and link not in self.urls_to_visit:
                                    self.urls_to_visit.append(link)

                    # 批次间延迟（可选）
                    if self.delay > 0 and self.urls_to_visit:
                        await asyncio.sleep(self.delay)

            print(f"\n[+] 异步爬取完成! 处理 {len(self.visited_urls)} 个页面")