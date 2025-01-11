revoke delete on table "public"."experiment_tag" from "anon";

revoke insert on table "public"."experiment_tag" from "anon";

revoke references on table "public"."experiment_tag" from "anon";

revoke select on table "public"."experiment_tag" from "anon";

revoke trigger on table "public"."experiment_tag" from "anon";

revoke truncate on table "public"."experiment_tag" from "anon";

revoke update on table "public"."experiment_tag" from "anon";

revoke delete on table "public"."experiment_tag" from "authenticated";

revoke insert on table "public"."experiment_tag" from "authenticated";

revoke references on table "public"."experiment_tag" from "authenticated";

revoke select on table "public"."experiment_tag" from "authenticated";

revoke trigger on table "public"."experiment_tag" from "authenticated";

revoke truncate on table "public"."experiment_tag" from "authenticated";

revoke update on table "public"."experiment_tag" from "authenticated";

revoke delete on table "public"."experiment_tag" from "service_role";

revoke insert on table "public"."experiment_tag" from "service_role";

revoke references on table "public"."experiment_tag" from "service_role";

revoke select on table "public"."experiment_tag" from "service_role";

revoke trigger on table "public"."experiment_tag" from "service_role";

revoke truncate on table "public"."experiment_tag" from "service_role";

revoke update on table "public"."experiment_tag" from "service_role";

revoke delete on table "public"."tags" from "anon";

revoke insert on table "public"."tags" from "anon";

revoke references on table "public"."tags" from "anon";

revoke select on table "public"."tags" from "anon";

revoke trigger on table "public"."tags" from "anon";

revoke truncate on table "public"."tags" from "anon";

revoke update on table "public"."tags" from "anon";

revoke delete on table "public"."tags" from "authenticated";

revoke insert on table "public"."tags" from "authenticated";

revoke references on table "public"."tags" from "authenticated";

revoke select on table "public"."tags" from "authenticated";

revoke trigger on table "public"."tags" from "authenticated";

revoke truncate on table "public"."tags" from "authenticated";

revoke update on table "public"."tags" from "authenticated";

revoke delete on table "public"."tags" from "service_role";

revoke insert on table "public"."tags" from "service_role";

revoke references on table "public"."tags" from "service_role";

revoke select on table "public"."tags" from "service_role";

revoke trigger on table "public"."tags" from "service_role";

revoke truncate on table "public"."tags" from "service_role";

revoke update on table "public"."tags" from "service_role";

alter table "public"."experiment_tag" drop constraint "experiment_tag_experiment_id_fkey";

alter table "public"."experiment_tag" drop constraint "experiment_tag_tag_id_fkey";

alter table "public"."tags" drop constraint "tags_name_key";

alter table "public"."experiment_tag" drop constraint "experiment_tag_pkey";

alter table "public"."tags" drop constraint "tags_pkey";

drop index if exists "public"."experiment_tag_pkey";

drop index if exists "public"."tags_name_key";

drop index if exists "public"."tags_pkey";

drop table "public"."experiment_tag";

drop table "public"."tags";

alter table "public"."experiment" add column "tags" text[];


