import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import fs from "fs";

export async function main() {

    if (fs.existsSync('/home/gordon/azure-hackathon/lib/serializedStore.json')) {
        const serializedStore = fs.readFileSync('/home/gordon/azure-hackathon/lib/serializedStore.json', 'utf8');
        const memoryVectors = JSON.parse(serializedStore);
        const vectorStore = new MemoryVectorStore(memoryVectors);
        const question = "What are the approaches to Task Decomposition?";
        const docs2 = await vectorStore.similaritySearch(question);
        console.log(docs2.length);
        return;
    }



    const loader = new CheerioWebBaseLoader(
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    );
    const docs = await loader.load();

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 0,
    });
    const allSplits = await textSplitter.splitDocuments(docs);
    console.log(allSplits.length);

    const embeddings = new OllamaEmbeddings({
        model: "llama3",
        baseUrl: "http://localhost:11434", // default value
        requestOptions: {
            useMMap: true, // use_mmap 1
            numThread: 6, // num_thread 6
            numGpu: 1, // num_gpu 1
        }
    });

    const vectorStore = await MemoryVectorStore.fromDocuments(
        allSplits,
        embeddings
    );

    const serializedStore = JSON.stringify(vectorStore.memoryVectors);
    fs.writeFileSync('/home/gordon/azure-hackathon/lib/serializedStore.json', serializedStore);

    const question = "What are the approaches to Task Decomposition?";
    const docs2 = await vectorStore.similaritySearch(question);
    console.log(docs2.length);
}