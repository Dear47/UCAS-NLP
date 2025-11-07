import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from corpus_crawler import CorpusCrawler

print("--- 开始爬取中文语料 ---")
chinese_start_urls = ["https://en.people.cn/"]
chinese_crawler = CorpusCrawler(
    start_urls=chinese_start_urls,
    output_dir="corpus_chinese",
    max_pages=50,  
    delay=1.5,
    lang='zh'
)
chinese_crawler.run()

print("\n" + "="*50 + "\n")

print("--- 开始爬取中文语料 ---")
chinese_start_urls = ["http://www.xinhuanet.com/"]
chinese_crawler = CorpusCrawler(
    start_urls=chinese_start_urls,
    output_dir="corpus_chinese",
    max_pages=50,  
    delay=1.5,
    lang='zh'
)
chinese_crawler.run()

print("\n" + "="*50 + "\n")

print("--- 开始爬取英文语料 ---")
english_start_urls = ["https://en.people.cn/"] 
english_crawler = CorpusCrawler(
    start_urls=english_start_urls,
    output_dir="corpus_english",
    max_pages=50,
    delay=2,
    lang='en'
)
english_crawler.run()

print("\n" + "="*50 + "\n")

print("--- 开始爬取英文语料 ---")
english_start_urls = ["https://english.news.cn/"] 
english_crawler = CorpusCrawler(
    start_urls=english_start_urls,
    output_dir="corpus_english",
    max_pages=50,
    delay=2,
    lang='en'
)
english_crawler.run()