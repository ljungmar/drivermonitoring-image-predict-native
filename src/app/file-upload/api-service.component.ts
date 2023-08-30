import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = 'http://192.168.32.14:8080';

  constructor(private http: HttpClient) { }

  uploadFile(formData: FormData) {
    return this.http.post<any>(`${this.baseUrl}/upload-file`, formData);
  }
}
