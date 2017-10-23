/*
 * App.cc
 *
 *  Created on: Mar 3, 2017
 *      Author: vance
 */

#include "App.h"
#include <cassert>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <vector>



App::App(appid_t appid, FILE* output, unsigned warp_size) : appid(appid), output(output) {
  shader_cycle_distro = (uint64_t*) calloc(warp_size + 3, sizeof(uint64_t));
  tlb_concurrent_total_time_app = (uint64_t*) calloc(200, sizeof(uint64_t));
}

App::~App() {
  fclose(output);
}

// Definition of static members
uint32_t appid_t::next_identifier = 666; // arbitrary
std::map<appid_t, App*> App::apps;
std::map<int, appid_t> App::sm_to_app;
std::map<int, appid_t> App::creation_index_to_app;
std::map<void*, appid_t> App::thread_id_to_app_id;
// special apps
App App::noapp(appid_t(), NULL, 0);
App App::pt_space(appid_t(), NULL, 0);
App App::mixapp(appid_t(), NULL, 0);
App App::prefrag(appid_t(), NULL, 0);

std::ostream& operator<<(std::ostream& os, const appid_t& appid) {
  os << appid.my_id;
  return os;
}

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
  return creation_index_to_app.find(i) != creation_index_to_app.end();
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

appid_t App::get_app_id_from_thread(void* tid) {
  return thread_id_to_app_id[tid];
}

App* App::get_app(appid_t app) {
  return App::apps.at(app);
}

appid_t App::register_app(int creation_index) {
  creation_index_to_app[creation_index] = appid_t();
  thread_id_to_app_id[(void*) pthread_self()] = creation_index_to_app[creation_index];
  std::cout << "Registering index " << creation_index << " as appID " <<
      creation_index_to_app[creation_index] << std::endl;
  return creation_index_to_app[creation_index];
}

appid_t App::create_app(appid_t my_id, FILE* output, unsigned warp_size) {
  static uint32_t addr_offset = 0;
  std::cout << "Creating app for appID " << my_id << std::endl;
  App::apps[my_id] = new App(my_id, output, warp_size);
  App::apps[my_id]->addr_offset = addr_offset++;
  return my_id;
}

std::map<appid_t, App*>& App::get_apps(void) {
  return App::apps;
}
