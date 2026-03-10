import csv
from collections import OrderedDict

with open("Results/e2eAgent/Results_as_csvs/cost_summary.csv") as f:
    rows = list(csv.DictReader(f))

hdr = "{:<20} {:<10} {:>12} {:>11} {:>12} {:>10} {:>12} {:>12} {:>13}"
row_fmt = "{:<20} {:<10} {:>12,} {:>10.3f}  {:>11.3f} {:>10.3f} ${:>11.4f} ${:>11.4f} ${:>12.4f}"
sub_fmt = "{:<20} {:<10} {:>12,} {:>10}  {:>11} {:>10} ${:>11.4f} ${:>11.4f} ${:>12.4f}"

print(hdr.format("Prompt Type", "LLM", "Tokens", "Input $/1M", "Output $/1M", "Avg $/1M", "Cost (low)", "Cost (avg)", "Cost (high)"))
print("-" * 115)

groups = OrderedDict()
for r in rows:
    groups.setdefault(r["prompt_type"], []).append(r)

grand_tokens = grand_low = grand_avg = grand_high = 0

for pt, items in groups.items():
    pt_tokens = pt_low = pt_avg = pt_high = 0
    for r in items:
        t = int(r["total_tokens"])
        low = float(r["cost_lower_input_only"])
        avg = float(r["cost_avg"])
        high = float(r["cost_upper_output_only"])
        print(row_fmt.format(pt, r["llm"], t, float(r["rate_input_per_1m"]), float(r["rate_output_per_1m"]), float(r["rate_avg_per_1m"]), low, avg, high))
        pt_tokens += t; pt_low += low; pt_avg += avg; pt_high += high
    print(sub_fmt.format("", "SUBTOTAL", pt_tokens, "", "", "", pt_low, pt_avg, pt_high))
    print("-" * 115)
    grand_tokens += pt_tokens; grand_low += pt_low; grand_avg += pt_avg; grand_high += pt_high

print(sub_fmt.format("GRAND TOTAL", "ALL", grand_tokens, "", "", "", grand_low, grand_avg, grand_high))
print("=" * 115)

print()
llm_fmt = "{:<10} {:>15,} ${:>13.4f} ${:>13.4f} ${:>13.4f}"
print("{:<10} {:>15} {:>14} {:>14} {:>14}".format("LLM", "Total Tokens", "Cost (low)", "Cost (avg)", "Cost (high)"))
print("-" * 70)
for llm in ["GPT5", "llama8B", "Mistral"]:
    lr = [r for r in rows if r["llm"] == llm]
    t = sum(int(r["total_tokens"]) for r in lr)
    low = sum(float(r["cost_lower_input_only"]) for r in lr)
    avg = sum(float(r["cost_avg"]) for r in lr)
    high = sum(float(r["cost_upper_output_only"]) for r in lr)
    print(llm_fmt.format(llm, t, low, avg, high))
print("-" * 70)
print(llm_fmt.format("ALL", grand_tokens, grand_low, grand_avg, grand_high))
print("=" * 70)
