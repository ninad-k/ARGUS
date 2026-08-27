"""Read-only analytics over historical picks and paper-trading outcomes.

Nothing here mutates a ``DailyPick``/``PaperOrder``/``PaperPosition`` row --
these modules only read the control-plane DB and ``BarStore`` to answer
"how did our picks actually do" questions (``argus.analysis.outcomes``) and
"which strategies/verdicts made paper money" questions
(``argus.analysis.attribution``).
"""
