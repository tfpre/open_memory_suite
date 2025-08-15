from open_memory_suite.benchmark.cost_model import CostModel, OperationType, ConcurrencyLevel


def test_cost_non_negative_and_additive():
    cm = CostModel()
    # store cost scales with tokens; retrieval scales with k
    c_store_small, _ = cm.predict(op=OperationType.STORE, adapter="memory_store", tokens=10, item_count=0)
    c_store_big, _ = cm.predict(op=OperationType.STORE, adapter="memory_store", tokens=1000, item_count=0)
    assert c_store_small >= 0 and c_store_big >= 0
    assert c_store_big >= c_store_small

    c_ret_k1, _ = cm.predict(op=OperationType.RETRIEVE, adapter="faiss_store", tokens=16, k=1, item_count=1000)
    c_ret_k10, _ = cm.predict(op=OperationType.RETRIEVE, adapter="faiss_store", tokens=16, k=10, item_count=1000)
    assert c_ret_k10 >= c_ret_k1
