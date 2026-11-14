SELECT
  count(DISTINCT user_id),
  --APPROX_COUNT_DISTINCT(user_id) AS unique_users,
  day
FROM events
GROUP BY day