WITH daily_user_stats AS (
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

ranked_users AS (
    SELECT
        event_date,
        user_id,
        total_events,
        exposures,
        clicks,

        ROUND(
            100.0 * clicks / NULLIF(exposures, 0),
            2
        ) AS ctr_percent,

        ROW_NUMBER() OVER (
            PARTITION BY event_date
            ORDER BY clicks DESC, user_id
        ) AS user_rank

    FROM daily_user_stats
)

SELECT
    event_date,
    user_rank,
    user_id,
    total_events,
    exposures,
    clicks,
    ctr_percent

FROM ranked_users

WHERE user_rank <= 20

ORDER BY
    event_date,
    user_rank;