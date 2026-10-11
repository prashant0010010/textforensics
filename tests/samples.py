"""Small hand-written (synthetic) sample texts for tests. They illustrate input kinds only."""
ACADEMIC = """Introduction

Urban green spaces influence residents' wellbeing in several measurable ways. Prior work, e.g. Smith et al. (2019), reports modest effects on stress. However, effect sizes vary widely across cities. In this paper we examine three mechanisms: exercise, social contact, and air quality.

Methods

We surveyed 412 residents of four districts. Participants answered questions about park use, perceived stress, and neighbourhood trust. Responses were analysed with mixed-effects models, which account for clustering by district. Missing data were handled by multiple imputation.

Results

Park use predicted lower stress (b = -0.21, p < 0.01). The association weakened after adjusting for income. Social contact explained a small share of the effect, while air quality showed no clear mediation. These findings suggest that access alone is not sufficient.

References

Smith, J., Lee, K., & Park, R. (2019). Green space and stress. Journal of Urban Health, 12(3), 45-60.
Jones, A. (2020). Parks and people. Urban Studies, 8(1), 1-20.
"""
TECHNICAL = """The scheduler maintains a priority queue of pending jobs. Each job has an identifier, a priority, and a deadline. When a worker becomes idle, the scheduler removes the job with the earliest deadline from the queue. If two jobs share a deadline, the one with the higher priority runs first. The algorithm runs in O(log n) time per operation. Memory use grows linearly with the number of pending jobs.

```
def next_job(queue):
    return heapq.heappop(queue)
```

| field | type |
| id | int |
| deadline | float |
"""
QUOTES = """The author begins with a claim.

"This is a long block quotation that deliberately contains many words so that the detector classifies it as a quotation block rather than ordinary prose."

> Another quoted passage in markdown style.

She then responds to the quotation in her own words, noting that the claim is only partly supported. The second sentence adds a caveat about the data."""
PUNCT = "Wait; what? Really!! Yes - truly: (a) one, (b) two, (c) three... \"quoted\" text, 'single' too; ok? Fine -- done."
UNICODE = "Café déjà vu naïve façade. Ünïcödé text with “curly quotes” and — dashes… 東京は日本の首都です。 Привет мир. Emoji 🙂 ok. e\u0301 combining mark."
SHORT = "Just a few words here."
SINGLE_PARA = ("The harbour was quiet that morning. Boats rocked gently against the pier, and gulls circled above the fish market. "
               "Nobody spoke much; the work of the day had not yet begun. By noon the quay would be crowded with buyers and sellers. "
               "For now, there was only the sound of water and rope. A cat watched from a coil of net.")
