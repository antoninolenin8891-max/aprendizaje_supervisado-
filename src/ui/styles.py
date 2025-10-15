def get_custom_css():
    return """
    <style>
    /* Todo tu CSS personalizado aquí */
    .main { background: #f8fafc; }
    .kpi-card { /* ... */ }
    .kpi-card-white { /* ... */ }
    /* ... resto del CSS ... */
    </style>
    """

def create_kpi_card(title, value, subtitle="", gradient=True):
    if gradient:
        return f"""
        <div class='kpi-card'>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-subtitle'>{subtitle}</div>
        </div>
        """
    else:
        return f"""
        <div class='kpi-card-white'>
            <div class='kpi-title-dark'>{title}</div>
            <div class='kpi-value-dark'>{value}</div>
            <div class='kpi-subtitle'>{subtitle}</div>
        </div>
        """