create table "public"."experiment_tag" (
    "experiment_id" uuid not null default gen_random_uuid(),
    "tag_id" uuid not null default gen_random_uuid(),
    "created_at" timestamp with time zone not null default now()
);


create table "public"."tags" (
    "id" uuid not null default gen_random_uuid(),
    "name" text,
    "created_at" timestamp with time zone not null default now()
);


CREATE UNIQUE INDEX experiment_tag_pkey ON public.experiment_tag USING btree (experiment_id, tag_id);

CREATE UNIQUE INDEX tags_name_key ON public.tags USING btree (name);

CREATE UNIQUE INDEX tags_pkey ON public.tags USING btree (id);

alter table "public"."experiment_tag" add constraint "experiment_tag_pkey" PRIMARY KEY using index "experiment_tag_pkey";

alter table "public"."tags" add constraint "tags_pkey" PRIMARY KEY using index "tags_pkey";

alter table "public"."experiment_tag" add constraint "experiment_tag_experiment_id_fkey" FOREIGN KEY (experiment_id) REFERENCES experiment(id) ON DELETE CASCADE not valid;

alter table "public"."experiment_tag" validate constraint "experiment_tag_experiment_id_fkey";

alter table "public"."experiment_tag" add constraint "experiment_tag_tag_id_fkey" FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE not valid;

alter table "public"."experiment_tag" validate constraint "experiment_tag_tag_id_fkey";

alter table "public"."tags" add constraint "tags_name_key" UNIQUE using index "tags_name_key";

grant delete on table "public"."experiment_tag" to "anon";

grant insert on table "public"."experiment_tag" to "anon";

grant references on table "public"."experiment_tag" to "anon";

grant select on table "public"."experiment_tag" to "anon";

grant trigger on table "public"."experiment_tag" to "anon";

grant truncate on table "public"."experiment_tag" to "anon";

grant update on table "public"."experiment_tag" to "anon";

grant delete on table "public"."experiment_tag" to "authenticated";

grant insert on table "public"."experiment_tag" to "authenticated";

grant references on table "public"."experiment_tag" to "authenticated";

grant select on table "public"."experiment_tag" to "authenticated";

grant trigger on table "public"."experiment_tag" to "authenticated";

grant truncate on table "public"."experiment_tag" to "authenticated";

grant update on table "public"."experiment_tag" to "authenticated";

grant delete on table "public"."experiment_tag" to "service_role";

grant insert on table "public"."experiment_tag" to "service_role";

grant references on table "public"."experiment_tag" to "service_role";

grant select on table "public"."experiment_tag" to "service_role";

grant trigger on table "public"."experiment_tag" to "service_role";

grant truncate on table "public"."experiment_tag" to "service_role";

grant update on table "public"."experiment_tag" to "service_role";

grant delete on table "public"."tags" to "anon";

grant insert on table "public"."tags" to "anon";

grant references on table "public"."tags" to "anon";

grant select on table "public"."tags" to "anon";

grant trigger on table "public"."tags" to "anon";

grant truncate on table "public"."tags" to "anon";

grant update on table "public"."tags" to "anon";

grant delete on table "public"."tags" to "authenticated";

grant insert on table "public"."tags" to "authenticated";

grant references on table "public"."tags" to "authenticated";

grant select on table "public"."tags" to "authenticated";

grant trigger on table "public"."tags" to "authenticated";

grant truncate on table "public"."tags" to "authenticated";

grant update on table "public"."tags" to "authenticated";

grant delete on table "public"."tags" to "service_role";

grant insert on table "public"."tags" to "service_role";

grant references on table "public"."tags" to "service_role";

grant select on table "public"."tags" to "service_role";

grant trigger on table "public"."tags" to "service_role";

grant truncate on table "public"."tags" to "service_role";

grant update on table "public"."tags" to "service_role";


