from datetime import date, timedelta
import adata


def get_last_trading_day():
    today = date.today()
    
    for i in range(10):
        check_date = today - timedelta(days=i)
        if is_trading_day(check_date):
            return check_date.strftime('%Y-%m-%d')
    
    return today.strftime('%Y-%m-%d')


def is_trading_day(check_date):
    try:
        test_data = adata.sentiment.hot.list_a_list_daily(check_date)
        return (hasattr(test_data, 'empty') and not test_data.empty) or len(test_data) > 0
    except Exception:
        return False