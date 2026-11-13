
#activated signal research candidates, not trading strategies in the portfolio

from foundation.data import asof_timestamp
from .core import TriggerSpec, TriggerRegistry

# name, source, kind, latency seconds, origin, hypothesis, unresolved research 'issue'
CANDIDATES = (
 ('insider_purchase','sec_edgar','form4',3600,'published_at','Voluntary insider purchases can reveal underpriced private conviction.','Verified Form 4 P/A transactions, issuer identity, repeat-filing controls and publication-to-fill costs.'),
 ('material_us_filing','sec_edgar','8k',900,'published_at','Material disclosures may diffuse into prices over time.','Classify item/content with information available at receipt; filing alone has no direction.'),
 ('uk_results_release','uk_rns','results',None,'published_at','Unexpected earnings or guidance can cause delayed repricing.','No RNS licence: authorised local/historical inputs only; PIT expectations and executable spreads.'),
 ('uk_regulatory_announcement','uk_rns','announcement',None,'published_at','Material UK announcements can trigger a fresh valuation decision.','No RNS licence: authorised local/historical inputs only; frozen materiality/direction rules.'),
 ('nsm_disclosure','fca_nsm','announcement',3600,'available_at','An archived regulated disclosure can open a slower research review.','NSM export receipt is availability; never equate it with original live publication.'),
 ('broker_headline','ibkr_news','headline',120,'published_at','An entitled relevant headline can change a predeclared decision state.','Provider coverage/rights, entity resolution and tested content classification.'),
 ('opening_gap','price_monitor','price_gap',60,'event_at','A discontinuous repricing can signal news assimilation or temporary imbalance.','Declare continuation versus reversal before evaluating; use tradable post-observation prices.'),
 ('range_breakout','price_monitor','range_breakout',60,'event_at','A move beyond a past-only range can indicate persistent demand.','Freeze lookback, volatility filter, false-breakout rule and exit.'),
 ('volatility_shock','price_monitor','volatility_shock',60,'event_at','An unusually large move can justify reducing portfolio risk.','Portfolio overlay versus alpha distinguished; exposure-matched controls required.'),
 ('macro_release','macro_calendar','scheduled_release',120,'event_at','A known release time opens a risk decision; surprises require separately received actuals.','Calendar vintages/revisions, actual receipt clocks and predeclared portfolio response.'),
 ('funding_extreme','crypto_metrics','funding_extreme',300,'event_at','Extreme derivative funding may indicate crowded positioning affecting spot.','Venue/period conventions; funding is information, never income on spot holdings.'),
 ('crypto_volatility','crypto_metrics','crypto_volatility',60,'event_at','A volatility shock can activate a spot exposure review.','Past-only baseline and comparison with exposure-matched nulls.'),
)


def catalogue():
    return [{**TriggerSpec(name,source,hypothesis,'max(published_at, received_at)',budget,(kind,),origin,
                          version=2 if source=='uk_rns' else 1,
                          latency_policy='next_xlon_open' if source=='uk_rns' else 'elapsed_seconds').__dict__,
             'signal_kind':'activated','status':'research_candidate','paper_only':True,'gate':gate}
            for name,source,kind,budget,origin,hypothesis,gate in CANDIDATES]


def default_registry():
    registry=TriggerRegistry()
    for row in catalogue():
        metadata={k:v for k,v in row.items() if k in TriggerSpec.__dataclass_fields__}
        name=row['name']
        if name=='insider_purchase': predicate=lambda event,at: event.payload.get('reported_purchase') is True
        elif name=='macro_release': predicate=lambda event,at: at>=event.available_at and at>=asof_timestamp(event.event_at)
        else: predicate=lambda event,at: True
        registry.register(TriggerSpec(**metadata),predicate)
    return registry
