SELECT id, host_name, start_time, end_time, category, description 
FROM public.metrics_exclusions 
WHERE host_name = %s 
ORDER BY id DESC;