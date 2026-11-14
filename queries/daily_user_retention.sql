WITH daily_user_activity AS (
    SELECT
        CAST(device_time AS DATE) AS event_date,
        user_id,

        COUNT(*) AS total_events,

        COUNT(*) FILTER (
            WHERE type = 'EXPOSED'
        ) AS exposures,

        COUNT(*) FILTER (
            WHERE type = 'CLICKED'
        ) AS clicks

    FROM events

    GROUP BY
        event_date,
        user_id
),

user_retention AS (
    SELECT
        current_day.event_date,
        current_day.user_id,
        current_day.total_events,
        current_day.exposures,
        current_day.clicks,

        CASE
            WHEN previous_day.user_id IS NOT NULL
            THEN 1
            ELSE 0
        END AS returned_from_previous_day

    FROM daily_user_activity current_day

    LEFT JOIN daily_user_activity previous_day
        ON current_day.user_id = previous_day.user_id
        AND previous_day.event_date =
            current_day.event_date - INTERVAL '1 day'
),

daily_stats AS (
    SELECT
        event_date,

        COUNT(*) AS daily_active_users,

        SUM(returned_from_previous_day) AS returning_users,

        COUNT(*) FILTER (
            WHERE clicks > 0
        ) AS users_who_clicked,

        SUM(total_events) AS total_events,
        SUM(exposures) AS total_exposures,
        SUM(clicks) AS total_clicks,

        ROUND(
            AVG(total_events),
            2
        ) AS avg_events_per_user

    FROM user_retention

    GROUP BY event_date
),

stats_with_previous_day AS (
    SELECT
        *,

        LAG(daily_active_users) OVER (
            ORDER BY event_date
        ) AS previous_day_active_users

    FROM daily_stats
)

SELECT
    event_date,
    daily_active_users,
    returning_users,
    users_who_clicked,

    ROUND(
        100.0 * returning_users /
        NULLIF(previous_day_active_users, 0),
        2
    ) AS retention_percent,

    total_events,
    total_exposures,
    total_clicks,

    ROUND(
        100.0 * total_clicks /
        NULLIF(total_exposures, 0),
        2
    ) AS ctr_percent,

    avg_events_per_user

FROM stats_with_previous_day

ORDER BY event_date;