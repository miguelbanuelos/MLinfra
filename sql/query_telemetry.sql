SELECT "timestamp" AS "Time", "cpu_max" AS "CPU_Usage" 
FROM public."Servers" 
WHERE "ServerName" = %s AND "timestamp" >= NOW() - (%s * INTERVAL '1 hour')
ORDER BY "timestamp" ASC;