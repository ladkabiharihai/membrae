import sys; sys.path.insert(0,"/home/ashish/Downloads/membrae")
import brain
P=lambda *a:print(*a,flush=True)
b = brain.Brain(learn=False); P("INIT OK (%.0fM)"%(b.n_params()/1e6))
known=["What is the capital of France","What is 2 plus 2","Who wrote Hamlet"]
nons =["What is the flarn of a quix","What is the zorbal index of Mars"]
# T1.1 semantic entropy (root fix): known should be LOW, nonsense HIGH
se=lambda q:b._semantic_entropy(f"<user> {q}? <assistant>",5,16)[0]
ks,ns=sum(map(se,known))/len(known),sum(map(se,nons))/len(nons)
P(f"T1.1 sem-entropy  known={ks:.2f} nonsense={ns:.2f}  sep={ns-ks:+.2f}  {'PASS' if ns>ks else 'FAIL'}")
# T1.2 rephrasing stability: known HIGH, nonsense LOW
rs=lambda q:b._rephrasing_stability(q)
kr,nr=sum(rs(q) for q in known)/len(known),sum(rs(q) for q in nons)/len(nons)
P(f"T1.2 rephrase-stab known={kr:.2f} nonsense={nr:.2f}  {'PASS' if kr>nr else 'FAIL'}")
# T3.3 router robustness (the misroute case + metas)
for q in ["What are black holes?","who are you?","how do you feel?","who wrote hamlet?"]:
    P(f"T3.3 route {q!r:28} -> {b._route_intent(q)}")
# T2.5 latent think
P("T2.5 latent_think ->", repr(b.latent_think("<user> What is the capital of France? <assistant>", steps=3, answer_n=20)[:70]))
# T2.6 background tick
P("T2.6 background_tick ->", b.background_tick())
# T2.1 workspace inject on/off + T2.2 goal vector
b.set_goal("learn about space"); b.broadcast("black holes")
P("T2.1 workspace_vector:", "ok" if b.workspace_vector() is not None else "None")
P("T2.1 inject-on  ->", repr(b.generate_with_workspace("<user> Tell me about space <assistant>", n=14, inject=True)[:50]))
P("T2.1 inject-off ->", repr(b.generate_with_workspace("<user> Tell me about space <assistant>", n=14, inject=False)[:50]))
P("T2.2 goal_vector:", "ok" if b.goal_vector() is not None else "None")
# T1.4 retrieval verification (internet)
sup,src=b.verify_against_source("Paris is the capital of France","capital of France")
P(f"T1.4 verify supported={sup} src={src}")
P("BATCH OK")
