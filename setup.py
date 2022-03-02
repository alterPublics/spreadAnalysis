import io
import os
import sys
import contextlib
from setuptools import setup,find_packages

@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)

def setup_package(pkg_name):

    root = os.path.abspath(os.path.dirname(__file__))

    with chdir(root):
        with io.open(os.path.join(root, "about.py"), encoding="utf8") as f:
            about = {}
            exec(f.read(), about)

    with io.open(os.path.join(root, "README.md"), encoding="utf8") as f:
        readme = f.read()

    setup(
        name= about["__title__"],
        version=   about["__version__"],
        author= about["__author__"],
        author_email=   about["__email__"],
        description=about["__summary__"],
        long_description=readme,
        url=about["__uri__"],
        packages=find_packages(),
        install_requires=[
            'pandas',
            'requests',
            'requests_oauthlib',
            'BeautifulSoup4',
            'bs4',
            'selenium',
            'openpyxl',
            'python-dotenv',
            'selenium',
            'timeout_decorator',
            'python-louvain',
            'fasttext',
            'langdetect',
            'telethon',
            'TikTokApi',
            'scipy',
            'nltk',
            'scikit-learn',
            'gensim',
            'seaborn',
            'matplotlib',
            'pymongo==3.9',
            'hdbscan',
            'garc',
            'newspaper3k',
            'networkit',
            'pandas-multiprocess'
        ],
        python_requires='>=3.8',
        )

if __name__ == "__main__":
    pkg_name = 'spreadAnalysis'
    setup_package(pkg_name)
