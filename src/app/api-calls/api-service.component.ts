import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { api_service_conf as conf} from './api-service-conf';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  constructor(private http: HttpClient) { }

  uploadFile(formData: FormData) {
    return this.http.post<any>(`${conf.baseUrl}/upload-file`, formData);
  }

  getFile() {
    const headers = new HttpHeaders({
      'X-AIO-Key': conf.adafruitKey
    });
    return this.http.get<any>(conf.adafruitUrl, { headers });
  }
}
