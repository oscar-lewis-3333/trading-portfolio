#dedicated to sending alerts when/if trading bot fails and for when the bot runs its trading process

import requests

def send_alert(subject, body, env_path=None):
    #send message of activity to discord server that only i am in

    import os 
    from dotenv import load_dotenv
    if env_path:
        load_dotenv(env_path)

    webhook_url = os.getenv('DISCORD_WEBHOOK')

    if not webhook_url:
        print("alerting not configured, skipping...")
        return False

    try:
        response = requests.post(webhook_url, json={"content": f"**{subject}**\n{body}"}, timeout=10)
        response.raise_for_status()
        print(f"Alert sent: {subject}")
        return True
    except Exception as e:
        print(f"Alert failed: {e}")
        return False

#create an alert message for when bot runs. categories are success, pending, or blocked.
def format_run_alert(result):
    if not isinstance(result, dict):
        raise TypeError("Bot result must be a dictionary")

    status = result.get("status")
    mode = result.get("mode", "unknown")
    if not isinstance(status, str) or not status:
        raise ValueError("Bot result has no valid status")
    if status == "holding_existing_basket": #if just holding, then no message necessary.
        return None

    success_subjects = {
        "order_submitted": "Trading Bot Order Submitted",
        "completed": "Trading Bot Rebalance Completed",
        "ready_to_prepare": "Trading Bot Dry Run Ready"
    }

    pending_statuses = {
        "pending_rebalance",
        "waiting_for_completion",
        "waiting_for_execution_window",
        "orders_in_progress",
        "waiting_for_reconciliation",
        "holdings_not_yet_reconciled",
        "waiting_for_fresh_prices",
        "submission_deferred"
    }

    blocked_statuses = {
        "blocked_open_orders",
        "execution_blocked",
        "outside_execution_window",
        "insufficient_cash"
    }

    if status in success_subjects: #determine which message to send based on bot's behaviour
        subject = success_subjects[status]
    elif status in pending_statuses:
        subject = "Trading Bot Orders Pending"
    elif status in blocked_statuses:
        subject = "Trading Bot Blocked"
    else:
        subject = "Trading Bot Status Requires Review"

    lines = [f"Status: {status}", f"Mode: {mode}"]

    pending = result.get("pending")

    if isinstance(pending, dict):
        rebalance_id = pending.get("rebalance_id")
        if rebalance_id:
            lines.append(f"Rebalance ID: {rebalance_id}")

        orders = pending.get("orders")
        if isinstance(orders, list):
            status_counts = {}

            for order in orders:
                order_status = str(order.get("status", "unknown"))
                status_counts[order_status] = status_counts.get(order_status, 0) + 1

            summary = ", ".join(f"{name}: {count}" for name, count in sorted(status_counts.items()))
            lines.append(f"Orders: {summary or 'none'}")

    if result.get("blockers"):
        lines.append(f"Blockers: {result['blockers']}")
    if result.get("mismatches"):
        lines.append(f"Position mismatches: {result['mismatches']}")
    if result.get("open_order_ids"):
        lines.append(f"Untracked open orders: {len(result['open_order_ids'])}")
    if result.get("message"):
        lines.append(str(result["message"]))

    return subject, "\n".join(lines)

#actually sending the alert, this is connected to run_bot_scheduled.py.
def send_run_alert(result, env_path=None):
    alert = format_run_alert(result)

    if alert is None:
        return False

    subject, body = alert
    return send_alert(subject=subject, body=body, env_path=env_path)
