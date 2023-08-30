import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'io.ionic.starter',
  appName: 'drivermonitoring_image_predict_native',
  webDir: 'www',
  server: {
    cleartext: true,
  },
  android: {
    allowMixedContent: true
  }
};

export default config;
