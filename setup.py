from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

exec(open("reflex_langgraph/version.py").read())

setup(
    name="reflex-langgraph",
    version=__version__,
    license="MIT",
    author="Jong Ha Shin",
    author_email="shinjh1206@gmail.com",
    keywords="reflex langgraph fastapi multiagent human-in-the-loop agents llm jobs",
    description="LangGraph multi-agent apps with a FastAPI backend and a Reflex UI: "
                "streaming chat, human approval, and long-running jobs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JjongX/reflex-langgraph",
    packages=find_packages(exclude=["examples*", "tests*"]),
    python_requires=">=3.11",
    install_requires=[
        "langgraph>=1.2",
        "langchain>=1.4",
        "fastapi>=0.115",
        "sse-starlette>=2.1",
        "pydantic>=2.7",
        "pyyaml>=6.0",
        "httpx>=0.27",
        "httpx-sse>=0.4",
    ],
    extras_require={
        "ui": ["reflex>=0.10,<0.11"],
        "google": ["langchain-google-genai>=4.0"],
        "openai": ["langchain-openai>=1.0"],
        "anthropic": ["langchain-anthropic>=1.0"],
        "server": ["uvicorn[standard]>=0.30"],
        "sqlite": ["langgraph-checkpoint-sqlite>=3.0", "aiosqlite>=0.20"],
        "s3": ["boto3>=1.34"],
        "dev": ["pytest>=8.0", "anyio>=4.0", "uvicorn[standard]>=0.30",
                "langgraph-checkpoint-sqlite>=3.0", "aiosqlite>=0.20"],
        "docs": ["sphinx>=7", "sphinx-rtd-theme", "autodocsumm", "myst-parser"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Framework :: FastAPI",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={"console_scripts": ["reflex-langgraph=reflex_langgraph.__main__:main"]},
    include_package_data=True,
)
