from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r'Text Splitters/research.pdf')

docs=loader.load()

text = """To effectively analyze and predict SLR, this study follows a
structured methodological approach, as illustrated in Figure 1. The
process begins with data collection and pre-processing, leveraging
historical MSL data from reputable sources, including satellite
altimetry records. The dataset is then subjected to resampling
and normalization to ensure consistency and eliminate noise.
Following data preparation, statistical and deep learning models
are developed and trained using both univariate and multivariate
approaches. The primary models include ARIMA and LSTM
networks, with further enhancements introduced through SE
blocks to improve the performance of LSTM. The evaluation
phase involves rigorous testing of the models and validation
against
historical
data.
The best-performing model is
subsequently used to forecast future SLR scenarios, and the
results are visualized through an interactive web platform,
offering dynamic representations of potentially inundated areas.
This comprehensive methodology ensures robust, data-driven
predictions that contribute to enhanced climate resilience planning."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

res=splitter.split_text(text) # for spliting text

#for doc
# res= splitter.split_documents(docs)
# print(res[1].page_content)

print(len(res))
print(res)