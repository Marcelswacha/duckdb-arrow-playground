SELECT
  count(DISTINCT user_id),
  CAST(device_time AS DATE) AS day
FROM events
GROUP BY day