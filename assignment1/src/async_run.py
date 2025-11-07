import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
import asyncio
from async_corpus_crawler import AsyncCorpusCrawler

async def main():
    chinese_start_urls = [
                        # "https://www.people.com.cn/",
                        # "https://www.xinhuanet.com/",
                        # "http://www.gov.cn",
                        # "https://news.cctv.com/",
                        # "http://www.cctv.com",
                        # "http://www.chinanews.com",
                        # "http://www.ce.cn/",
                        # "http://news.sina.com.cn",
                        # "http://news.163.com",
                        # "http://news.sohu.com",
                        # "https://www.douban.com/"
                        # "https://www.zhihu.com/"
                        # "https://blog.csdn.net/",
                        # "https://spaces.ac.cn/"
                        # "https://zh.wikisource.org/wiki/Wikisource:%E9%A6%96%E9%A1%B5"
                        # "https://marxists.org.cn/chinese/"
                        ]
    
    english_start_urls = [
                        # "https://en.people.cn/",
                        # "https://english.news.cn/",
                        # "http://www.chinadaily.com.cn",
                        # "https://www.cgtn.com/",
                        # "https://www.globaltimes.cn/",
                        # "https://plato.stanford.edu/",
                        # "https://arxiv.org/",
                        # "https://doaj.org/",
                        # "https://www.worldbank.org/en/research"
                        # "https://gutendex.com"
                        # "https://www.python.org/"
                        ]
    
    for ch_url in chinese_start_urls: 
        print(f"--- 开始爬取中文语料 {ch_url}---")
        ch_crawler = AsyncCorpusCrawler(
            start_urls=[ch_url],
            output_dir="./corpus_ch",
            max_pages=200,
            delay=1,
            lang='zh',
            max_concurrent=4  # 高并发可能会使爬虫访问被阻止
        )
        await ch_crawler.run()
        print("\n" + "="*50 + "\n")

    for en_url in english_start_urls:
        print(f"--- 开始爬取英文语料 {en_url}---")
        en_crawler = AsyncCorpusCrawler(
            start_urls=[en_url],
            output_dir="./corpus_en",
            max_pages=200,
            delay=1,
            lang='en',
            max_concurrent=4
        )
        await en_crawler.run()
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())