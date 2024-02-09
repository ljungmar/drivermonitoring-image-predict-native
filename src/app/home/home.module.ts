import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IonicModule } from '@ionic/angular';
import { FormsModule } from '@angular/forms';
import { HomePage } from './home.page';

import { HomePageRoutingModule } from './home-routing.module';
import { FileUploadComponent } from '../file-upload/file-upload.component';
import { FileCamUploadComponent } from '../file-cam-upload/file-cam-upload.component';
import { FileItemComponent } from '../file-item/file-item.component';
import { DesktopViewComponent } from '../desktop-view/desktop-view.component';


@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    HomePageRoutingModule
  ],
  declarations: [HomePage, FileUploadComponent, FileItemComponent, FileCamUploadComponent, DesktopViewComponent]
})
export class HomePageModule {}
