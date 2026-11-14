WITH content_stats AS (
    SELECT
        content_id,

        COUNT(*) AS total_events,

        COUNT(*) FILTER (
            WHERE type = 'EXPOSED'
        ) AS exposures,

        COUNT(*) FILTER (
            WHERE type = 'CLICKED'
        ) AS clicks,

        COUNT(DISTINCT user_id) AS unique_users

    FROM events

    GROUP BY content_id
),

top_content AS (
    SELECT
        content_id,
        total_events,
        exposures,
        clicks,
        unique_users,

        ROUND(
            100.0 * clicks / NULLIF(exposures, 0),
            2
        ) AS ctr_percent

    FROM content_stats

    WHERE exposures >= 10

    ORDER BY
        clicks DESC,
        content_id

    LIMIT 100
),

country_language_stats AS (
    SELECT
        e.content_id,
        e.country,
        e.language,

        COUNT(*) AS total_events,

        COUNT(*) FILTER (
            WHERE e.type = 'EXPOSED'
        ) AS exposures,

        COUNT(*) FILTER (
            WHERE e.type = 'CLICKED'
        ) AS clicks

    FROM events e

    INNER JOIN top_content t
        ON e.content_id = t.content_id

    GROUP BY
        e.content_id,
        e.country,
        e.language
),

ranked_segments AS (
    SELECT
        *,

        ROW_NUMBER() OVER (
            PARTITION BY content_id
            ORDER BY clicks DESC, country, language
        ) AS segment_rank

    FROM country_language_stats
)

SELECT
    t.content_id,
    t.total_events,
    t.exposures,
    t.clicks,
    t.unique_users,
    t.ctr_percent,

    s.country,
    s.language,
    s.exposures AS segment_exposures,
    s.clicks AS segment_clicks,

    ROUND(
        100.0 * s.clicks / NULLIF(s.exposures, 0),
        2
    ) AS segment_ctr_percent

FROM top_content t

INNER JOIN ranked_segments s
    ON t.content_id = s.content_id

WHERE s.segment_rank <= 3

ORDER BY
    t.clicks DESC,
    t.content_id,
    s.segment_rank;