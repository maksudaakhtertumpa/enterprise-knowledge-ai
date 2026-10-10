"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api";

export interface Load<T> {
  data: T | undefined;
  error: ApiError | undefined;
  loading: boolean;
  reload: () => void;
}

export function asApiError(e: unknown): ApiError {
  return e instanceof ApiError ? e : new ApiError(0, "client_error", String(e));
}

/**
 * Fetch on mount, whenever `key` changes, and on reload(). Keeps the last data
 * while a new request is in flight; a slow old response never overwrites a newer one.
 */
export function useLoad<T>(fn: () => Promise<T>, key: string): Load<T> {
  const fnRef = useRef(fn);
  useEffect(() => {
    fnRef.current = fn;
  });
  const [nonce, setNonce] = useState(0);
  const token = `${key}#${nonce}`;
  const [state, setState] = useState<{ token?: string; data?: T; error?: ApiError }>({});

  useEffect(() => {
    let current = true;
    fnRef.current().then(
      (data) => current && setState({ token, data }),
      (e) => current && setState((s) => ({ token, data: s.data, error: asApiError(e) })),
    );
    return () => {
      current = false;
    };
  }, [token]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data: state.data, error: state.error, loading: state.token !== token, reload };
}
