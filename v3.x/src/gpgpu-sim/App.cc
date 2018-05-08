/*
 * App.cc
 *
 *  Created on: Mar 3, 2017
 *      Author: vance
 */

#include "App.h"
#include <map>
#include <cassert>
#include <vector>
#include <stdio.h>

App::App(appid_t appid, FILE* output, unsigned warp_size) : appid(appid), output(output) {
  shader_cycle_distro = (uint64_t*) calloc(warp_size + 3, sizeof(uint64_t));
  // no reason this has to be here... I just have no idea what's going on.
  tlb_concurrent_total_time_app = (uint64_t*) calloc(200, sizeof(uint64_t));
  // I do not know what the 200 is. It's TLB-related. TODO define a constant.
}

App::~App() {
  fclose(output);
}

// Definition of static members
std::map<appid_t, App*> App::apps;
std::map<int, appid_t> App::sm_to_app;
std::map<int, appid_t> App::creation_index_to_app;
appid_t App::next_app_id = 666; // a random, nonzero  application identifier

const std::vector<int> App::get_app_sms(appid_t appid) {
  std::vector<int> sms;
  for (std::map<int, appid_t>::const_iterator i = App::sm_to_app.cbegin();
      i != App::sm_to_app.cend(); i++) {
    if (i->second == appid) {
      sms.push_back(i->first);
    }
  }
  return sms;
}

bool App::is_registered(int i)
{
  bool return_val = (creation_index_to_app.find(i) != creation_index_to_app.end());
  printf("Check if app is registered, return val = %d\n", return_val);
  return return_val;
}

/**
 * Assigns each sm in sms to appid.
 * Old assignments to appid are retained; if you want to unset them, call get_app_sms first
 * and set_app_sms to an invalid appid on the sms you want to unset.
 *
 * This function is best called while reassigning all appids to new sms.
 */
void App::set_app_sms(appid_t appid, std::vector<int>& sms) {
  for (std::vector<int>::const_iterator sm = sms.cbegin(); sm != sms.cend(); sm++) {
    App::sm_to_app[*sm] = appid;
  }
}

appid_t App::get_app_id(int creation_index) {
  return creation_index_to_app.at(creation_index);
}

appid_t App::get_app_id_from_sm(int sm_number) {
  return sm_to_app.at(sm_number);
}

App* App::get_app(appid_t app) {
  return App::apps.at(app);
}

appid_t App::register_app(int creation_index) {
  appid_t my_id = next_app_id;
  printf("Registering index = %d, as appID = %d\n",creation_index, my_id);
  creation_index_to_app[creation_index] = next_app_id++;
  return my_id;
}

appid_t App::create_app(appid_t my_id, FILE* output, unsigned warp_size) {
  printf("Creating app for appID = %d\n", my_id);
  App::apps[my_id] = new App(my_id, output, warp_size);
  return my_id;
}

std::map<appid_t, App*>& App::get_apps(void) {
  return App::apps;
}
