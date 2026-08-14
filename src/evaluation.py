def hit_at_k(retrieved_records, expected_doc_uuid):
    return int(
        any(
            record.get("doc_uuid") == expected_doc_uuid
            for record in retrieved_records
        )
    )


def reciprocal_rank(retrieved_records, expected_doc_uuid):
    for rank, record in enumerate(retrieved_records, start=1):
        if record.get("doc_uuid") == expected_doc_uuid:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(values):
    if not values:
        return 0.0

    return sum(values) / len(values)