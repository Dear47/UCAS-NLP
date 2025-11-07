import os
import requests
import time
import hashlib
import re
from collections import deque
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

class CorpusCrawler:
    """
    一个基础的网络爬虫, 用于爬取指定网站的中文或英文语料
    它会遵守 robots.txt 规则, 进行速率限制, 并对文本进行清洗
    """
    
    def __init__(self, start_urls, output_dir, max_pages=50, delay=1, lang='en'):
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
        
        self.urls_to_visit = deque(start_urls)
        self.visited_urls = set()
        self.robot_parsers = {}
        self.user_agent = 'MyCorpusCrawler/1.0 (+https://github.com/your-repo)'
        self.headers = {'User-Agent': self.user_agent}
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[*] 输出目录: {self.output_dir}")
        print(f"[*] 爬虫模式: {self.lang}")

    def _get_robot_parser(self, url):
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
        rp = self._get_robot_parser(url)
        if rp:
            return rp.can_fetch(self.user_agent, url)
        return True

    def _clean_chinese(self, text):
        """
        清洗中文文本: 只保留中文字符, 其他替换为空格。
        """
        # 匹配所有非中文字符 (\u4e00-\u9fff), 将其替换为单个空格
        pattern = re.compile(r'[^\u4e00-\u9fff]')
        cleaned_text = re.sub(pattern, '', text)
        
        # 将连续的多个空白(空格、换行等)合并为单个空格
        cleaned_text = re.sub(r'\s+', '', cleaned_text).strip()
        return cleaned_text

    def _clean_english(self, text):
        """
        清洗英文文本: 只保留26个英文字母, 其他替换为空格。
        """
        # 匹配所有非英文字母 (a-z, A-Z)将, 其替换为单个空格
        pattern = re.compile(r'[^a-zA-Z]')
        cleaned_text = re.sub(pattern, '', text)
        
        # 将连续的多个空白(空格、换行等)合并为单个空格
        cleaned_text = re.sub(r'\s+', '', cleaned_text).strip()
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
            print(f"[*] 页面无有效语料，跳过保存: {url}")
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

    def run(self):
        """
        启动爬虫主循环。
        """
        print(f"[*] 开始爬取... 最大页面数: {self.max_pages}")
        
        while self.urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = self.urls_to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
                
            if not self._can_fetch(current_url):
                print(f"[-] robots.txt 禁止爬取: {current_url}")
                self.visited_urls.add(current_url)
                continue
            
            print(f"[*] G: ({len(self.visited_urls) + 1}/{self.max_pages}): {current_url}")
            
            try:
                time.sleep(self.delay)
                
                response = requests.get(current_url, headers=self.headers, timeout=10)
                
                self.visited_urls.add(current_url)
                
                if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                    
                    # 修复乱码
                    response.encoding = response.apparent_encoding
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = self._extract_text(soup)
                    
                    self._save_text(current_url, text)
                    
                    self._find_links(current_url, soup)
                    
                else:
                    print(f"[!] 无法获取 HTML (状态码: {response.status_code}): {current_url}")

            except requests.RequestException as e:
                print(f"[!] 请求异常 {current_url}: {e}")
                self.visited_urls.add(current_url) 
            except Exception as e:
                print(f"[!] 处理时发生未知错误 {current_url}: {e}")
                self.visited_urls.add(current_url)

        print(f"\n[+] 爬取完成。总共访问了 {len(self.visited_urls)} 个页面。")