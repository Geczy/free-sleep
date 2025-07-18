import { useSettings } from '@api/settings.ts';
import { useIsFetching } from '@tanstack/react-query';
import moment from 'moment-timezone';
import type React from 'react';
import { useEffect } from 'react';
import { create } from 'zustand';
import { getFieldFromIndexedDB, updateFieldInIndexedDB } from './indexedDB.ts';

export type Side = 'left' | 'right';

type AppState = {
  isUpdating: boolean;
  setIsUpdating: (isUpdating: boolean) => void;
  side: Side;
  setSide: (side: Side) => void;
};

// Create Zustand store
export const useAppStore = create<AppState>((set) => ({
  isUpdating: false,
  setIsUpdating: (isUpdating: boolean) => set({ isUpdating }),
  side: 'left',
  setSide: (side: Side) => {
    set({ side });
    updateFieldInIndexedDB('side', side)
      .then(() => {})
      .catch((error) => {
        console.error(error);
      });
  },
}));

// AppStoreProvider to sync Zustand with react-query's isFetching
export function AppStoreProvider({ children }: React.PropsWithChildren) {
  // Exclude base status query from global fetching count since it's background polling
  const totalFetching = useIsFetching();
  const baseFetching = useIsFetching({ queryKey: ['baseStatus'] });
  const isFetching = totalFetching - baseFetching > 0;
  const { data: settings } = useSettings();

  const { side, setSide } = useAppStore();
  useEffect(() => {
    if (!settings) return;
    // @ts-ignore
    moment.tz.setDefault(settings.timeZone);
  }, [settings]);

  // Directly update the store's state
  useEffect(() => {
    useAppStore.setState({ isUpdating: isFetching });
  }, [isFetching]);

  useEffect(() => {
    getFieldFromIndexedDB('side')
      .then((resp) => {
        if (resp) {
          setSide(resp);
        } else {
          return updateFieldInIndexedDB('side', side);
        }
      })
      .then(() => {})
      .catch((error) => {
        console.error('Error initializing IndexedDB:', error);
      });
  }, []);

  return <>{children}</>;
}
