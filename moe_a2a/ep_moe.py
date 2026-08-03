import torch
import torch.distributed as dist

def expert_ffn(token, gate, down):
        return torch.nn.functional.silu(token @ gate) @ down

def build_dispatch_plan(topk_ids, experts_per_rank, world_size, rank):
    """Route tokens to destination ranks. Return(send_counts, flat_toks, flat_metas)"""
    send_counts = [0] * world_size
    send_toks = [[] for _ in range(world_size)]
    send_metas = [[] for _ in range(world_size)]

    for i, expert_ids in enumerate(topk_ids.tolist()):
        for j, expert_id in enumerate(expert_ids):
            dst = expert_id // experts_per_rank
            send_counts[dst] += 1
            send_toks[dst].append(i)
            send_metas[dst].append((rank, expert_id, i, j))

    flat_toks = [r for dst in range(world_size) for r in send_toks[dst]]
    flat_metas = [r for dst in range(world_size) for r in send_metas[dst]]

    return send_counts, flat_toks, flat_metas

def dispatch(tokens, send_counts, flat_toks, flat_metas, world_size):
    send_counts_t = torch.tensor(send_counts, dtype=torch.int32)
    recv_counts_t = torch.empty(world_size, dtype=torch.int32)
    dist.all_to_all_single(recv_counts_t, send_counts_t)

    total_recvs = int(torch.sum(recv_counts_t))
    embed_dim = tokens.shape[-1]

    send_toks_t = tokens[flat_toks]
    recv_toks_t = torch.empty(total_recvs, embed_dim, dtype=torch.float32)
    dist.all_to_all_single(recv_toks_t, send_toks_t, recv_counts_t.tolist(), send_counts_t.tolist())

    send_metas_t = torch.tensor(flat_metas, dtype=torch.int32)
    recv_metas_t = torch.empty(total_recvs, 4, dtype=torch.int32)
    dist.all_to_all_single(recv_metas_t, send_metas_t, recv_counts_t.tolist(), send_counts_t.tolist())

    return recv_toks_t, recv_metas_t, recv_counts_t

def run_experts(recv_toks_t, recv_metas_t, experts_gate, experts_down, my_experts):
    expert_out = torch.empty_like(recv_toks_t)
    eids = recv_metas_t[:, 1]
    for eid in my_experts:
        mask = (eids == eid)
        if mask.any():
            expert_out[mask] = expert_ffn(recv_toks_t[mask], experts_gate[eid], experts_down[eid])

    return expert_out

def combine(expert_out_t, flat_metas, send_counts, recv_counts_t, n_tokens, topk_w, rank):
    embed_dim = expert_out_t.shape[-1]
    total_recv = n_tokens * topk_w.shape[-1]
    expert_in_t = torch.empty(total_recv, embed_dim, dtype=torch.float32)
    dist.all_to_all_single(expert_in_t, expert_out_t, send_counts, recv_counts_t.tolist())

    result = torch.zeros(n_tokens, embed_dim, dtype=torch.float32)
    for i in range(expert_in_t.shape[0]):
        tok = expert_in_t[i]
        _, _, tok_id, w_id = flat_metas[i]

        result[tok_id] += topk_w[tok_id, w_id] * tok

    return result

def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    n_tokens = 10
    d_token, d_model = 16, 32
    top_k = 4
    experts = 16
    experts_per_rank = experts // world_size

    torch.manual_seed(42)
    expert_gate = [torch.randn(d_token, d_model, dtype=torch.float32) for _ in range(experts)]
    expert_down = [torch.randn(d_model, d_token, dtype=torch.float32) for _ in range(experts)]
    my_experts = range(experts_per_rank * rank, experts_per_rank * (rank + 1))

    torch.manual_seed(rank)
    tokens = torch.randn(n_tokens, d_token, dtype=torch.float32)
    router_w = torch.randn(d_token, experts, dtype=torch.float32)
    logits = tokens @ router_w
    topk_w, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)

    refer = torch.zeros(n_tokens, d_token, dtype=torch.float32)
    for i in range(n_tokens):
        token = tokens[i]
        for j in range(top_k):
            eid = topk_ids[i, j]
            refer[i] += topk_w[i, j] * expert_ffn(token, expert_gate[eid], expert_down[eid])

    send_counts, flat_toks, flat_metas = build_dispatch_plan(topk_ids, experts_per_rank, world_size, rank)
    recv_toks_t, recv_metas_t, recv_counts_t = dispatch(tokens, send_counts, flat_toks, flat_metas, world_size)
    expert_out_t = run_experts(recv_toks_t, recv_metas_t, expert_gate, expert_down, my_experts)
    result = combine(expert_out_t, flat_metas, send_counts, recv_counts_t, n_tokens, topk_w, rank)

    ok = torch.allclose(result, refer, rtol=1e-5, atol=1e-5)
    ok_t = torch.tensor([1 if ok else 0])
    dist.all_reduce(ok_t)                          # sum across ranks
    if rank == 0:
        print(f"PASS: {int(ok_t)}/{world_size} ranks correct")

if __name__ == "__main__":
    main()