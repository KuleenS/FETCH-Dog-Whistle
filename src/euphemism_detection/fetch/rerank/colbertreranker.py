from collections import Counter

from typing import List

from colbert import Indexer, Searcher

from colbert.data import Queries, Collection, Ranking

from colbert.infra import Run, RunConfig, ColBERTConfig

from colbert.infra.provenance import Provenance

import torch

from tqdm import tqdm


class COLBERTReranker:

    def __init__(
        self,
        corpus: List[str],
        colbert_path: str,
        min_ranker: bool = False,
        max_len: int = 256,
        nbits: int = 2,
    ):
        self.collection = Collection(data=corpus)

        self.colbert_path = colbert_path

        self.min_ranker = min_ranker

        self.max_len = max_len

        self.nbits = nbits

    def dense_search(self, searcher, Q: torch.Tensor, k=10, filter_fn=None, pids=None):
        if k <= 10:
            if searcher.config.ncells is None:
                self.configure(ncells=1)
            if searcher.config.centroid_score_threshold is None:
                searcher.configure(centroid_score_threshold=0.5)
            if searcher.config.ndocs is None:
                searcher.configure(ndocs=256)
        elif k <= 100:
            if searcher.config.ncells is None:
                searcher.configure(ncells=2)
            if searcher.config.centroid_score_threshold is None:
                searcher.configure(centroid_score_threshold=0.45)
            if searcher.config.ndocs is None:
                searcher.configure(ndocs=1024)
        else:
            if searcher.config.ncells is None:
                searcher.configure(ncells=4)
            if searcher.config.centroid_score_threshold is None:
                searcher.configure(centroid_score_threshold=0.4)
            if searcher.config.ndocs is None:
                searcher.configure(ndocs=max(k * 4, 4096))

        pids, scores = searcher.ranker.rank(
            searcher.config, Q, filter_fn=filter_fn, pids=pids
        )

        if self.min_ranker:
            return pids[:-k], list(range(1, k + 1)), scores[:-k]
        else:
            return pids[:k], list(range(1, k + 1)), scores[:k]

    def _search_all_Q(self, searcher, queries, Q, k, filter_fn=None, qid_to_pids=None):
        qids = list(queries.keys())

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        all_scored_pids = [
            list(
                zip(
                    *self.dense_search(
                        searcher,
                        Q[query_idx : query_idx + 1],
                        k,
                        filter_fn=filter_fn,
                        pids=qid_to_pids[qid],
                    )
                )
            )
            for query_idx, qid in tqdm(enumerate(qids))
        ]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = "Searcher::search_all"
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def search_all(
        self,
        searcher,
        queries,
        k=10,
        filter_fn=None,
        full_length_search=False,
        qid_to_pids=None,
    ):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(
            searcher, queries, Q, k, filter_fn=filter_fn, qid_to_pids=qid_to_pids
        )

    def rerank(
        self,
        queries: List[str],
        dogwhistles: List[str],
        top_k_colbert: int,
        top_k_count: int,
    ):

        queries_dict = dict(zip(range(len(queries)), queries))

        queries = Queries(data=queries_dict)

        top_reranked = Counter()

        with Run().context(RunConfig(nranks=1, experiment="reranker")):
            config = ColBERTConfig(doc_maxlen=self.max_len, nbits=self.nbits)

            indexer = Indexer(checkpoint=self.colbert_path, config=config)
            indexer.index(
                name="dogwhistles", collection=self.collection, overwrite=True
            )

        with Run().context(RunConfig(experiment="reranker")):
            searcher = Searcher(index="dogwhistles")

        rankings = self.search_all(searcher, queries, k=top_k_colbert).todict()

        for ranking_key in rankings:
            docs = rankings[ranking_key]

            for doc in docs:
                top_reranked[doc[1]] += 1

        most_common_indexes = [x[0] for x in top_reranked.most_common(top_k_count)]

        return [dogwhistles[x] for x in most_common_indexes]
