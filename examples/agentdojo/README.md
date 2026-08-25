# AgentDojo banking-suite demo (R49b)

Shows the R49b suite wiring end-to-end against the real
[AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark: the
banking suite's tools are registered through `gate_suite`, untrusted
readers (`read_file`, `get_most_recent_transactions`) taint the session,
and `policies/agentdojo.yaml` denies external sinks once the session is
tainted.

```bash
pip install agentdojo   # or: pip install agent-policy-gateway[agentdojo]
python -m examples.agentdojo
```

Expected output — an untainted `send_money` and the read-only
`user_task_1` pass, while the injected incoming-transaction subject
(`injection_task_0`'s exfiltration) is refused at `send_money` with rule
`deny-untrusted-to-send_money`, and the environment shows no transaction
to the attacker's IBAN. The entry point exits non-zero on any other
outcome, so it can serve as a CI sanity check wherever the benchmark is
installed.
